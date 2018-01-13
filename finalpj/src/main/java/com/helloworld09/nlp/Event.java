package com.helloworld09.nlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.Quadruple;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;

import java.io.Serializable;

class Event implements Serializable {
    private IndexedWord verb;
    private IndexedWord protagonist;

    protected enum EventRelation {
        SUBJ, OBJ, UNKNOWN
    }

    protected EventRelation relation;
    private Logger logger = Logger.getLogger(Event.class);

    public Event() {
    }

    public Event(IndexedWord verb, IndexedWord protagonist, String reln) {
        this.verb = verb;
        this.protagonist = protagonist;
        this.relation = convertRelation(reln);
    }

    public Event(IndexedWord verb, IndexedWord protagonist, GrammaticalRelation reln) {
        this.verb = verb;
        this.protagonist = protagonist;
        this.relation = convertRelation(reln.getShortName());
    }

    public EventRelation convertRelation(String relation) {
        EventRelation reln;

        if (relation.contains("subj")) {
            reln = EventRelation.SUBJ;
        } else if (relation.contains("obj")) {
            reln = EventRelation.OBJ;
        } else {
            logger.error("Error event construction! relation = " + relation);
            reln = EventRelation.UNKNOWN;
        }
        return reln;
    }

    @Override
    public String toString() {
        return verb.value() + "\t" + protagonist.value() + "\t" + StringUtils.lowerCase(relation.toString());
    }

    public String toString(boolean lemma) {
        if (lemma)
            return verb.value() + "\t" + verb.get(CoreAnnotations.LemmaAnnotation.class) + "\t" + protagonist.value() + "\t" + StringUtils.lowerCase(relation.toString());
        else
            return toString();
    }

}

class ComplexEvent extends Event implements Serializable {
    private IndexedWord verb;
    private IndexedWord subject;
    private IndexedWord object;
    private IndexedWord preposition;

    public ComplexEvent(Quadruple<IndexedWord, IndexedWord, IndexedWord, IndexedWord> objects, String reln) {
        verb = objects.first();
        subject = objects.second();
        object = objects.third();
        preposition = objects.fourth();
        relation = convertRelation(reln);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(verb.value());
        sb.append(String.valueOf(verb.get(CoreAnnotations.LemmaAnnotation.class)));
        sb.append(String.valueOf(subject.value()));
        sb.append(String.valueOf(object.value()));
        sb.append(String.valueOf(preposition.value()));
        sb.append(StringUtils.lowerCase(relation.toString()));
        return String.join("\t", sb);
    }

}
