package com.helloworld09.nlp.util;

import org.apache.log4j.BasicConfigurator;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation;
import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;


import java.util.*;

public class Pipeline {

    private StanfordCoreNLP pipeline;

    public Pipeline(String property) {
        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        props.setProperty("annotators", property);
        pipeline = new StanfordCoreNLP(props);
    }

    public Annotation annotate(String text) {

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);

        return document;
    }

    public static void main(String[] args) {
        BasicConfigurator.configure();

        String property = "tokenize, ssplit, pos, lemma, ner, parse, dcoref";
        Pipeline ppl = new Pipeline(property);

        String text = "This project aims to predict what will happen next given a sequence of events in history";
        Annotation document = ppl.annotate(text);
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);

        for (CoreMap sentence : sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                // this is the text of the token
                String word = token.get(TextAnnotation.class);
                // this is the POS tag of the token
                String pos = token.get(PartOfSpeechAnnotation.class);
                // this is the NER label of the token
                String ne = token.get(NamedEntityTagAnnotation.class);

                System.out.println(word + "\t" + pos + "\t" + ne);
            }

            // this is the parse tree of the current sentence
            Tree tree = sentence.get(TreeAnnotation.class);
            System.out.println(tree);

            // this is the Stanford dependency graph of the current sentence
            SemanticGraph dependencies = sentence.get(EnhancedDependenciesAnnotation.class);
            System.out.println(dependencies);
        }
        Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);
        System.out.println(graph);
    }

}