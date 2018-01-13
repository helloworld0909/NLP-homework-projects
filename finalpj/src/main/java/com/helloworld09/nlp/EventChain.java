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

import edu.stanford.nlp.util.Quadruple;
import org.apache.log4j.BasicConfigurator;
import org.json.JSONObject;
import org.json.JSONArray;
import org.apache.log4j.Logger;
import org.apache.commons.lang3.StringUtils;

import com.helloworld09.nlp.util.Pipeline;
import com.helloworld09.nlp.util.Interpreter;
import com.helloworld09.nlp.Event.*;



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

    private void getEdgeMaps(List<CoreMap> sentences,
                             GrammaticalRelation[] filters,
                             Map<IntPair, List<SemanticGraphEdge>> edgeMapByVerbPosition,
                             Map<IntPair, List<SemanticGraphEdge>> edgeMapByProtaPosition){

        // Indexed from 1
        for (int sentNum = 1; sentNum <= sentences.size(); sentNum++) {
            CoreMap sentence = sentences.get(sentNum - 1);
            SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation.class);
            Map<String, List<SemanticGraphEdge>> relations = Interpreter.filterRelnsByDep(dependencies, filters);
            for (GrammaticalRelation filter : filters) {
                String shortName = filter.getShortName();
                for (SemanticGraphEdge edge : relations.get(shortName)) {
                    Pair<IndexedWord, IndexedWord> protaAndVerb = getNounAndVerb(edge);
                    IndexedWord protagonist = protaAndVerb.first();
                    IndexedWord verb = protaAndVerb.second();

                    IntPair verbPosition = new IntPair(sentNum, verb.index());
                    IntPair protagonistPosition = new IntPair(sentNum, protagonist.index());

                    edgeMapByVerbPosition.putIfAbsent(verbPosition, new LinkedList<>());
                    edgeMapByProtaPosition.putIfAbsent(protagonistPosition, new LinkedList<>());
                    edgeMapByVerbPosition.get(verbPosition).add(edge);
                    edgeMapByProtaPosition.get(protagonistPosition).add(edge);

                }
            }
        }
    }

    private List<List<Event>> extractEventChains(String text, GrammaticalRelation[] filters) {

        Map<Integer, List<Event>> eventChainsMap = new HashMap<>();
        Map<Integer, List<Event>> detailEventChainsMap = new HashMap<>();

        Annotation document = pipeline.annotate(text);

        Map<Integer, CorefChain> graph = document.get(CorefCoreAnnotations.CorefChainAnnotation.class);

        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

        Map<IntPair, List<SemanticGraphEdge>> edgeMapByVerbPosition = new LinkedHashMap<>();
        Map<IntPair, List<SemanticGraphEdge>> edgeMapByProtaPosition = new LinkedHashMap<>();
        getEdgeMaps(sentences, filters, edgeMapByVerbPosition, edgeMapByProtaPosition);

        for (Map.Entry<IntPair, List<SemanticGraphEdge>> entry : edgeMapByProtaPosition.entrySet()) {
            for (CorefChain chain : graph.values()) {
                IntPair protaPosition = entry.getKey();
                if (chain.getMentionMap().get(protaPosition) != null) {
                    int chainID = chain.getChainID();
                    List<SemanticGraphEdge> protaRelatedEdges = entry.getValue();
                    for (SemanticGraphEdge protaRelatedEdge : protaRelatedEdges) {
                        Pair<IndexedWord, IndexedWord> protaAndVerb = getNounAndVerb(protaRelatedEdge);
                        IndexedWord protagonist = protaAndVerb.first();
                        IndexedWord verb = protaAndVerb.second();

                        eventChainsMap.putIfAbsent(chainID, new ArrayList<>());
                        String chainRelationStr = protaRelatedEdge.getRelation().getShortName();
                        Event event = new Event(verb, protagonist, chainRelationStr);
                        eventChainsMap.get(chainID).add(event);

                        IndexedWord subject = null, object = null, prepositionalEntity = null;
                        switch (Event.convertRelation(protaRelatedEdge).toString()) {
                            case "SUBJ": {
                                subject = protagonist;
                                break;
                            }
                            case "OBJ":
                                object = protagonist;
                                break;
                            default:
                                logger.error("edge = " + protaRelatedEdge);
                        }

                        IntPair verbPosition = new IntPair(protaPosition.get(0), verb.index());
                        List<SemanticGraphEdge> verbRelatedEdges = edgeMapByVerbPosition.get(verbPosition);

                        // Add additional information
                        if (verbRelatedEdges.size() > 1) {
                            for (SemanticGraphEdge relatedEdge : verbRelatedEdges) {
                                String relationStr = relatedEdge.getRelation().getShortName();

                                IndexedWord indexedWord = getNounAndVerb(relatedEdge).first();

                                if (!indexedWord.equals(protagonist)) {
                                    if (!relationStr.equals(chainRelationStr)) {
                                        switch (Event.convertRelationToString(chainRelationStr)) {
                                            case "SUBJ": {
                                                object = indexedWord;
                                                break;
                                            }
                                            case "OBJ": {
                                                subject = indexedWord;
                                                break;
                                            }
                                        }

                                    }
                                    if (relationStr.equals("PREP") & !indexedWord.get(CoreAnnotations.NamedEntityTagAnnotation.class).equals("O")) {
                                        prepositionalEntity = indexedWord;
                                    }
                                }
                            }
                        }

                        Quadruple<IndexedWord, IndexedWord, IndexedWord, IndexedWord> params = new Quadruple<>(verb, subject, object, prepositionalEntity);
                        Event detailEvent = new ComplexEvent(params, chainRelationStr);
                        detailEventChainsMap.putIfAbsent(chainID, new ArrayList<>());
                        detailEventChainsMap.get(chainID).add(detailEvent);
                    }
                    break;
                }
            }
        }
        System.out.println(detailEventChainsMap);
        logger.debug("Finish extracting event chain");
        return new ArrayList<>(eventChainsMap.values());
    }

    private Pair<IndexedWord, IndexedWord> getNounAndVerb(SemanticGraphEdge edge) {
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
                new GrammaticalRelation(Language.Any, "nsubjpass", "SubjectPass", null),
                new GrammaticalRelation(Language.Any, "pobj", "ObjectPreposition", null),
                new GrammaticalRelation(Language.Any, "prep", "Preposition", null),
                new GrammaticalRelation(Language.Any, "prepc", "PrepositionClausal", null),
        };
        String inputDirPath = "data/";
        File inputDir = new File(inputDirPath);
        if (inputDir.exists()) {
            String fileNameList[] = inputDir.list();
            for (String fileName : fileNameList) {
                if (fileName.endsWith(".json")) {
                    eventChainBuilder.buildEventChain("data/" + fileName, "output/" + fileName.split("\\.")[0] + ".txt", filters);
                }
            }
        } else
            logger.error("Input dir path not found");
    }
}
