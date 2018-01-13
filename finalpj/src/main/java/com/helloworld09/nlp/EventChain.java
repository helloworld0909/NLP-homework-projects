package com.helloworld09.nlp;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.io.FileWriter;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.international.Language;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;

import org.apache.log4j.BasicConfigurator;
import org.json.JSONObject;
import org.json.JSONArray;
import org.apache.log4j.Logger;
import org.apache.commons.lang3.StringUtils;

import com.helloworld09.nlp.util.Pipeline;
import com.helloworld09.nlp.util.Interpreter;


public class EventChain {

    private static Logger logger;
    private Pipeline pipeline;

    public EventChain(String property) {

        logger = Logger.getLogger(EventChain.class);
        pipeline = new Pipeline(property);
    }

    public void buildEventChain(String filename, String outputFileName, GrammaticalRelation[] filters) {
        try {
            String content = readFile(filename, StandardCharsets.UTF_8);
            JSONArray obj = new JSONArray(content);
            FileWriter outputFile = new FileWriter(outputFileName);

            for (int i = 0; i < obj.length(); i++) {
                JSONObject docObj = obj.getJSONObject(i);
                String docID = docObj.getString("id");

                if (!docObj.isNull("paragraphs")) {
                    JSONArray paragraphs = docObj.getJSONArray("paragraphs");
                    StringBuilder docBuilder = new StringBuilder();
                    for (Object p : paragraphs) {
                        p = StringUtils.trim((String) p);
                        docBuilder.append(p);
                        docBuilder.append('\n');
                    }
                    String text = docBuilder.toString();
                    List<List<Event>> eventChains = extractEventChains(text, filters);
                    int chainIdx = 0;
                    for (List<Event> chain : eventChains) {
                        outputFile.write(docID + "\t" + chainIdx + "\n");
                        for (Event e : chain) {
                            outputFile.write(e.toString(true) + "\n");
                        }
                        outputFile.write("\n");
                        chainIdx++;
                    }
                    outputFile.flush();
                }
            }
            outputFile.close();

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            logger.info("Finish building event train on " + filename);
        }
    }

    private List<List<Event>> extractEventChains(String text, GrammaticalRelation[] filters) {

        Map<Integer, List<Event>> eventChainsMap = new HashMap<>();
        Annotation document = pipeline.annotate(text);

        Map<Integer, CorefChain> graph = document.get(CorefCoreAnnotations.CorefChainAnnotation.class);

        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

        Map<IntPair, SemanticGraphEdge> edgeMapByVerbPosition = new HashMap<>();
        Map<IntPair, SemanticGraphEdge> edgeMapByProtaPosition = new HashMap<>();
        // Indexed from 1
        for (int sentNum = 1; sentNum <= sentences.size(); sentNum++) {
            CoreMap sentence = sentences.get(sentNum - 1);
            SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation.class);
            Map<String, List<SemanticGraphEdge>> relations = Interpreter.filterRelnsByDep(dependencies, filters);
            for (GrammaticalRelation filter : filters) {
                String shortName = filter.getShortName();
                for (SemanticGraphEdge edge : relations.get(shortName)) {
                    Pair<IndexedWord, IndexedWord> protaAndVerb = getProtaAndVerb(edge);
                    IndexedWord protagonist = protaAndVerb.first();
                    IndexedWord verb = protaAndVerb.second();

                    IntPair verbPosition = new IntPair(sentNum, verb.index());
                    IntPair protagonistPosition = new IntPair(sentNum, protagonist.index());
                    edgeMapByVerbPosition.put(verbPosition, edge);
                    edgeMapByProtaPosition.put(protagonistPosition, edge);
                }
            }
        }
        for (Map.Entry<IntPair, SemanticGraphEdge> entry: edgeMapByProtaPosition.entrySet()){
            for (CorefChain chain : graph.values()) {
                if (chain.getMentionsWithSameHead(entry.getKey()) != null) {
                    int chainID = chain.getChainID();
                    SemanticGraphEdge edge = entry.getValue();
                    Pair<IndexedWord, IndexedWord> protaAndVerb = getProtaAndVerb(edge);
                    IndexedWord protagonist = protaAndVerb.first();
                    IndexedWord verb = protaAndVerb.second();

                    eventChainsMap.putIfAbsent(chainID, new ArrayList<>());
                    Event event = new Event(verb, protagonist, edge.getRelation().getShortName());
                    eventChainsMap.get(chainID).add(event);
                }
            }
        }

        logger.debug("Finish extracting event chain");
        return new ArrayList<>(eventChainsMap.values());
    }

    private Pair<IndexedWord, IndexedWord> getProtaAndVerb(SemanticGraphEdge edge) {
        if (!edge.getSource().tag().startsWith("V")) {
            return new Pair<>(edge.getSource(), edge.getTarget());
        } else {
            return new Pair<>(edge.getTarget(), edge.getSource());
        }
    }

    private static String readFile(String path, Charset encoding) throws IOException {
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }

    public static void main(String[] args) {
        BasicConfigurator.configure();
        EventChain eventChainBuilder = new EventChain("tokenize, ssplit, pos, lemma, ner, parse, mention, coref");
        GrammaticalRelation[] filters = {
                new GrammaticalRelation(Language.Any, "nsubj", "Subject", null),
                new GrammaticalRelation(Language.Any, "dobj", "Object", null),
        };
        String inputDirPath = "data/";
        File inputDir = new File(inputDirPath);
        if (inputDir.exists()) {
            String fileNameList[] = inputDir.list();
            for (String fileName : fileNameList) {
                if(fileName.endsWith(".json")) {
                    eventChainBuilder.buildEventChain("data/" + fileName, "output/" + fileName.split("\\.")[0] + ".txt", filters);
                }
            }
        }
        else
            logger.error("Input dir path not found");
    }
}
