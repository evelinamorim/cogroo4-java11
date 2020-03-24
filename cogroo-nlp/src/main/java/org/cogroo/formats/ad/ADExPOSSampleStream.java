/**
 * Copyright (C) 2012 cogroo <cogroo@cogroo.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.cogroo.formats.ad;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import opennlp.tools.formats.ad.ADSentenceStream;
import opennlp.tools.formats.ad.ADSentenceStream.Sentence;
import opennlp.tools.formats.ad.ADSentenceStream.SentenceParser.Leaf;
import opennlp.tools.formats.ad.ADSentenceStream.SentenceParser.Node;
import opennlp.tools.formats.ad.ADSentenceStream.SentenceParser.TreeElement;
import opennlp.tools.postag.POSSample;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

import com.google.common.base.Strings;

/**
 * <b>Note:</b> Do not use this class, internal use only!
 */
public class ADExPOSSampleStream implements ObjectStream<POSSample> {

  private final ObjectStream<ADSentenceStream.Sentence> adSentenceStream;
  private boolean expandME;
  private boolean isIncludeFeatures;
  private boolean additionalContext;
  
  // this is used to control changing aspas representation, some sentences we keep as original, others we change to " 
  private int callsCount = 0;
  
  private static final Pattern hyphenPattern = Pattern.compile("((\\p{L}+)-$)|(^-(\\p{L}+)(.*))|((\\p{L}+)-(\\p{L}+)(.*))");

  /**
   * Creates a new {@link POSSample} stream from a line stream, i.e.
   * {@literal {@link ObjectStream}< {@link String}>}, that could be a
   * {@link PlainTextByLineStream} object.
   * 
   * @param lineStream
   *          a stream of lines as {@link String}
   * @param expandME
   *          if true will expand the multiword expressions, each word of the
   *          expression will have the POS Tag that was attributed to the
   *          expression plus the prefix B- or I- (CONLL convention)
   * @param includeFeatures
   *          if true will combine the POS Tag with the feature tags
   * @param additionalContext
   *          to set additional context
   */
  public ADExPOSSampleStream(ObjectStream<String> lineStream, boolean expandME,
      boolean includeFeatures, boolean additionalContext) {
    this.adSentenceStream = new ADSentenceStream(lineStream);
    this.expandME = expandME;
    this.isIncludeFeatures = includeFeatures;
    this.additionalContext = additionalContext;
  }


  public POSSample read() throws IOException {
    
    callsCount++;
    
    Sentence paragraph;
    while ((paragraph = this.adSentenceStream.read()) != null) {
      Node root = paragraph.getRoot();
      List<String> sentence = new ArrayList<String>();
      List<String> tags = new ArrayList<String>();
      List<String> contractions = new ArrayList<String>();
      List<String> prop = new ArrayList<String>();
      process(root, sentence, tags, contractions, prop);

      if (sentence.size() != contractions.size()
          || sentence.size() != prop.size()) {
        throw new IllegalArgumentException(
            "There must be exactly same number of tokens and additional context!");
      }

      if(this.additionalContext) {
        String[][] ac = new String[2][sentence.size()];
        // line 0: contractions
        // line 1: props
        for (int i = 0; i < sentence.size(); i++) {
          if (contractions.get(i) != null) {
            ac[0][i] = contractions.get(i);
          }
          if (prop.get(i) != null) {
            ac[1][i] = prop.get(i);
          }
        }
        // System.out.println();
        return new POSSample(sentence, tags, ac);
      } else {
        return new POSSample(sentence, tags);
      }
    }
    return null;
  }

  private void process(Node node, List<String> sentence, List<String> tags,
      List<String> con, List<String> prop) {
    if (node != null) {
      for (TreeElement element : node.getElements()) {
        if (element.isLeaf()) {
          processLeaf((Leaf) element, sentence, tags, con, prop);
        } else {
          process((Node) element, sentence, tags, con, prop);
        }
      }
    }
  }

  
  private void processLeaf(Leaf leaf, List<String> sentence, List<String> tags,
      List<String> con, List<String> prop) {
    if (leaf != null) {
      String lexeme = leaf.getLexeme();
      
      // this will change half of the aspas 
      if("«".equals(lexeme) || "»".equals(lexeme)) {
        if(callsCount % 2 == 0) {
          lexeme = "\"";
        }
      }
      String tag = leaf.getFunctionalTag();

      String contraction = null;
      if (leaf.getSecondaryTag() != null) {
        if (leaf.getSecondaryTag().contains("<sam->")) {
          contraction = "B";
        } else if (leaf.getSecondaryTag().contains("<-sam>")) {
          contraction = "E";
        }
      }

      if (tag == null) {
        tag = lexeme;
      }

      if (isIncludeFeatures && leaf.getMorphologicalTag() != null) {
        tag += " " + leaf.getMorphologicalTag();
      }
      
      tag = tag.replaceAll("\\s+", "=");
      
      if (tag == null)
        tag = lexeme;

      if (expandME && lexeme.contains("_")) {
        StringTokenizer tokenizer = new StringTokenizer(lexeme, "_");

        if ("prop".equals(tag)) {
          sentence.add(lexeme);
          tags.add(tag);
          con.add(null);
          prop.add("P");
        } else if (tokenizer.countTokens() > 0) {
          List<String> toks = new ArrayList<String>(tokenizer.countTokens());
          List<String> tagsWithCont = new ArrayList<String>(
              tokenizer.countTokens());
          toks.add(tokenizer.nextToken());
          tagsWithCont.add("B-" + tag);
          while (tokenizer.hasMoreTokens()) {
            toks.add(tokenizer.nextToken());
            tagsWithCont.add("I-" + tag);
          }
          if (contraction != null) {
            con.addAll(Arrays.asList(new String[toks.size() - 1]));
            con.add(contraction);
          } else {
            con.addAll(Arrays.asList(new String[toks.size()]));
          }

          sentence.addAll(toks);
          tags.addAll(tagsWithCont);
          prop.addAll(Arrays.asList(new String[toks.size()]));
        } else {
          sentence.add(lexeme);
          tags.add(tag);
          prop.add(null);
          con.add(contraction);
        }

      } else if(lexeme.contains("-") && lexeme.length() > 1) {
        Matcher matcher = hyphenPattern.matcher(lexeme);

        String firstTok = null;
        String hyphen = "-";
        String secondTok = null;
        String rest = null;

        if (matcher.matches()) {
          if (matcher.group(1) != null) {
            firstTok = matcher.group(2);
          } else if (matcher.group(3) != null) {
            secondTok = matcher.group(4);
            rest = matcher.group(5);
          } else if (matcher.group(6) != null) {
            firstTok = matcher.group(7);
            secondTok = matcher.group(8);
            rest = matcher.group(9);
          } else {
            throw new IllegalStateException("wrong hyphen pattern");
          }

          if (!Strings.isNullOrEmpty(firstTok)) {
            sentence.add(firstTok);
            tags.add(tag);
            prop.add(null);
            con.add(contraction);
          }
          
          if (!Strings.isNullOrEmpty(hyphen)) {
            sentence.add(hyphen);
            tags.add("-");
            prop.add(null);
            con.add(contraction);
          }
          if (!Strings.isNullOrEmpty(secondTok)) {
            sentence.add(secondTok);
            tags.add(tag);
            prop.add(null);
            con.add(contraction);
          }
          if (!Strings.isNullOrEmpty(rest)) {
            sentence.add(rest);
            tags.add(tag);
            prop.add(null);
            con.add(contraction);
          }
        } else {
          sentence.add(lexeme);
          tags.add(tag);
          prop.add(null);
          con.add(contraction);
        }
      } else {
        tag = addGender(tag, leaf.getMorphologicalTag());
        
        sentence.add(lexeme);
        tags.add(tag);
        prop.add(null);
        con.add(contraction);
      }
    }
  }
  
  private static final Pattern GENDER_M = Pattern.compile(".*\\bM\\b.*"); 
  private static final Pattern GENDER_F = Pattern.compile(".*\\bF\\b.*"); 
  private static final Pattern GENDER_N = Pattern.compile(".*\\bM/F\\b.*"); 

  private String addGender(String tag, String morphologicalTag) {
    if(("n".equals(tag) || "art".equals(tag)) && morphologicalTag != null) {
      if(GENDER_N.matcher(morphologicalTag).matches()) {
        //tag = tag + "n";
      } else if(GENDER_M.matcher(morphologicalTag).matches()) {
        tag = tag + "m";
      } else if(GENDER_F.matcher(morphologicalTag).matches()) {
        tag = tag + "f";
      } 
      
    }
    return tag;
  }


  public void reset() throws IOException, UnsupportedOperationException {
    adSentenceStream.reset();
  }

  public void close() throws IOException {
    adSentenceStream.close();
  }
}
