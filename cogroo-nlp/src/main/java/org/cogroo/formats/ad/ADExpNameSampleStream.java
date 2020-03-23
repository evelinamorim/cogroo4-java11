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
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import opennlp.tools.formats.ad.ADSentenceStream;
import opennlp.tools.formats.ad.ADSentenceStream.Sentence;
import opennlp.tools.formats.ad.ADSentenceStream.SentenceParser.Leaf;
import opennlp.tools.formats.ad.ADSentenceStream.SentenceParser.Node;
import opennlp.tools.formats.ad.ADSentenceStream.SentenceParser.TreeElement;
import opennlp.tools.formats.ad.PortugueseContractionUtility;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;

/**
 * Parser for Floresta Sita(c)tica Arvores Deitadas corpus, output to for the
 * Portuguese NER training.
 * <p>
 * The data contains common multiword expressions. The categories are:<br>
 * intj, spec, conj-s, num, pron-indef, n, prop, adj, prp, adv
 * <p>
 * Data can be found on this web site:<br>
 * http://www.linguateca.pt/floresta/corpus.html
 * <p>
 * Information about the format:<br>
 * Susana Afonso.
 * "Árvores deitadas: Descrição do formato e das opções de análise na Floresta Sintáctica"
 * .<br>
 * 12 de Fevereiro de 2006.
 * http://www.linguateca.pt/documentos/Afonso2006ArvoresDeitadas.pdf
 * <p>
 * Detailed info about the NER tagset:
 * http://beta.visl.sdu.dk/visl/pt/info/portsymbol.html#semtags_names
 * <p>
 * <b>Note:</b> Do not use this class, internal use only!
 */
public class ADExpNameSampleStream implements ObjectStream<NameSample> {

  private ObjectStream<ADSentenceStream.Sentence> adSentenceStream = null;

  /**
   * To keep the last left contraction part
   */
  private String leftContractionPart = null;

  /**
   * The tags we are looking for
   */
  private Set<String> tags;

  private final boolean useAdaptativeFeatures;

  /**
   * Creates a new {@link NameSample} stream from a line stream, i.e.
   * {@literal {@link ObjectStream}< {@link String}>}, that could be a
   * {@link PlainTextByLineStream} object.
   * 
   * @param lineStream
   *          a stream of lines as {@link String}
   * @param tags
   *          the tags we are looking for, or null for all
   * @param useAdaptativeFeatures flag to use adaptative features
   */
  public ADExpNameSampleStream(ObjectStream<String> lineStream,
      Set<String> tags, boolean useAdaptativeFeatures) {
    this.adSentenceStream = new ADSentenceStream(lineStream);
    this.tags = tags;
    this.useAdaptativeFeatures = useAdaptativeFeatures;
  }

  /**
   * Creates a new {@link NameSample} stream from a {@link InputStream}
   * 
   * @param in
   *          the Corpus {@link InputStream}
   * @param charsetName
   *          the charset of the Arvores Deitadas Corpus
   * @param tags
   *          the tags we are looking for, or null for all
   * @param useAdaptativeFeatures
   *          flat to use or not adaptative features
   */
  public ADExpNameSampleStream(InputStreamFactory in, String charsetName,
                               Set<String> tags, boolean useAdaptativeFeatures) {
    this.useAdaptativeFeatures = useAdaptativeFeatures;
    try {
      this.adSentenceStream = new ADSentenceStream(new PlainTextByLineStream(
          in, charsetName));
      this.tags = tags;
    } catch (UnsupportedEncodingException e) {
      // UTF-8 is available on all JVMs, will never happen
      throw new IllegalStateException(e);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  int textID = -1;

  public NameSample read() throws IOException {

    Sentence paragraph;
    while ((paragraph = this.adSentenceStream.read()) != null) {

      boolean clearData = false;

      if (useAdaptativeFeatures) {
        int currentTextID = getTextID(paragraph);
        if (currentTextID != textID) {
          clearData = true;
          textID = currentTextID;
        }
      } else {
        clearData = true;
      }

      Node root = paragraph.getRoot();
      List<String> sentence = new ArrayList<String>();
      List<Span> names = new ArrayList<Span>();
      process(root, sentence, names);

      return new NameSample(sentence.toArray(new String[sentence.size()]),
          names.toArray(new Span[names.size()]), clearData);
    }
    return null;
  }

  enum Type {
    ama, cie, lit
  }

  private Type corpusType = null;

  private Pattern metaPattern;

  private int textIdMeta2 = -1;
  private String textMeta2 = "";

  private int getTextID(Sentence paragraph) {

    String meta = paragraph.getMetadata();

    if (corpusType == null) {
      if (meta.startsWith("LIT")) {
        corpusType = Type.lit;
        metaPattern = Pattern.compile("^([a-zA-Z\\-]+)(\\d+).*?p=(\\d+).*");
      } else if (meta.startsWith("CIE")) {
        corpusType = Type.cie;
        metaPattern = Pattern.compile("^.*?source=\"(.*?)\".*");
      } else { // ama
        corpusType = Type.ama;
        metaPattern = Pattern.compile("^(?:[a-zA-Z\\-]*(\\d+)).*?p=(\\d+).*");
      }
    }

    if (corpusType.equals(Type.lit)) {
      Matcher m2 = metaPattern.matcher(meta);
      if (m2.matches()) {
        String textId = m2.group(1);
        if (!textId.equals(textMeta2)) {
          textIdMeta2++;
          textMeta2 = textId;
        }
        return textIdMeta2;
      } else {
        throw new RuntimeException("Invalid metadata: " + meta);
      }
    } else if (corpusType.equals(Type.cie)) {
      Matcher m2 = metaPattern.matcher(meta);
      if (m2.matches()) {
        String textId = m2.group(1);
        if (!textId.equals(textMeta2)) {
          textIdMeta2++;
          textMeta2 = textId;
        }
        return textIdMeta2;
      } else {
        throw new RuntimeException("Invalid metadata: " + meta);
      }
    } else if (corpusType.equals(Type.ama)) {
      Matcher m2 = metaPattern.matcher(meta);
      if (m2.matches()) {
        return Integer.parseInt(m2.group(1));
        // currentPara = Integer.parseInt(m.group(2));
      } else {
        throw new RuntimeException("Invalid metadata: " + meta);
      }
    }

    return 0;
  }

  /**
   * Recursive method to process a node in Arvores Deitadas format.
   * 
   * @param node
   *          the node to be processed
   * @param sentence
   *          the sentence tokens we got so far
   * @param names
   *          the names we got so far
   */
  private void process(Node node, List<String> sentence, List<Span> names) {
    if (node != null) {
      for (TreeElement element : node.getElements()) {
        if (element.isLeaf()) {
          processLeaf((Leaf) element, sentence, names);
        } else {
          process((Node) element, sentence, names);
        }
      }
    }
  }

  /**
   * Process a Leaf of Arvores Detaitadas format
   * 
   * @param leaf
   *          the leaf to be processed
   * @param sentence
   *          the sentence tokens we got so far
   * @param names
   *          the names we got so far
   */
  private void processLeaf(Leaf leaf, List<String> sentence, List<Span> names) {

    if (leaf != null && leftContractionPart == null) {

      String namedEntityTag = null;
      int startOfNamedEntity = -1;

      String leafTag = leaf.getSecondaryTag();

      if (leafTag != null) {
        if (leafTag.contains("<sam->")) {
          String[] lexemes = leaf.getLexeme().split("_");
          if (lexemes.length > 1) {
            for (int i = 0; i < lexemes.length - 1; i++) {
              sentence.add(lexemes[i]);
            }
          }
          leftContractionPart = lexemes[lexemes.length - 1];
          return;
        }
        if (leaf.getLexeme().contains("_") && leaf.getLexeme().length() > 3) {
          String tag = leaf.getFunctionalTag();
          if (tags != null) {
            if (tags.contains(tag)) {
              namedEntityTag = leaf.getFunctionalTag();
            }
          } else {
            namedEntityTag = leaf.getFunctionalTag();
          }
        }
      }

      if (namedEntityTag != null) {
        startOfNamedEntity = sentence.size();
      }

      sentence.addAll(Arrays.asList(leaf.getLexeme().split("_")));

      if (namedEntityTag != null) {
        names
            .add(new Span(startOfNamedEntity, sentence.size(), namedEntityTag));
      }

    } else {
      // will handle the contraction
      String tag = leaf.getSecondaryTag();
      String right = leaf.getLexeme();
      if (tag != null && tag.contains("<-sam>")) {
        right = leaf.getLexeme();
        String c = PortugueseContractionUtility.toContraction(
            leftContractionPart, right);

        if (c != null) {
          sentence.add(c);
        } else {
          System.err.println("missing " + leftContractionPart + " + " + right);
          sentence.add(leftContractionPart);
          sentence.add(right);
        }

      } else {
        System.err.println("unmatch" + leftContractionPart + " + " + right);
      }
      leftContractionPart = null;
    }

  }

  public void reset() throws IOException, UnsupportedOperationException {
    adSentenceStream.reset();
  }

  public void close() throws IOException {
    adSentenceStream.close();
  }

}
