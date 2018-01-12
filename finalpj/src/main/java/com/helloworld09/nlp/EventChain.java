package com.helloworld09.nlp;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.international.Language;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.CoreMap;

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

    public void buildEventChain(String filename, GrammaticalRelation[] filters) {
        try {
            String content = readFile(filename, StandardCharsets.UTF_8);
            JSONArray obj = new JSONArray(content);
            for (int i = 0; i < obj.length(); i++) {
                JSONObject docObj = obj.getJSONObject(i);

                if (!docObj.isNull("paragraphs")) {
                    JSONArray paragraphs = docObj.getJSONArray("paragraphs");
                    StringBuilder docBuilder = new StringBuilder();
                    for (Object p : paragraphs) {
                        p = StringUtils.trim((String) p);
                        docBuilder.append(p);
                        docBuilder.append('\n');
                    }
                    String text = docBuilder.toString();
                    Annotation document = pipeline.annotate(text);
                    List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
                    for (CoreMap sentence : sentences) {
                        SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation.class);
                        Map relations = Interpreter.filterRelnsByDep(dependencies, filters);
                    }
                    Map<Integer, CorefChain> graph = document.get(CorefCoreAnnotations.CorefChainAnnotation.class);
                    System.out.println(graph);

                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            logger.info("Finish building event chain");
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
        eventChainBuilder.buildEventChain("data/nyt_eng_199407.json", filters);
    }
}
