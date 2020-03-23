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
package org.cogroo.cmdline.featurizer;

import java.io.OutputStream;

import org.cogroo.tools.featurizer.FeatureSample;
import org.cogroo.tools.featurizer.FeaturizerEvaluationMonitor;

import opennlp.tools.cmdline.EvaluationErrorPrinter;
import opennlp.tools.util.eval.EvaluationMonitor;

/**
 * A default implementation of {@link EvaluationMonitor} that prints to an
 * output stream.
 * 
 */
public class FeaturizerEvaluationErrorListener extends
    EvaluationErrorPrinter<FeatureSample> implements
    FeaturizerEvaluationMonitor {

  /**
   * Creates a listener that will print to System.err
   */
  public FeaturizerEvaluationErrorListener() {
    super(System.err);
  }

  /**
   * Creates a listener that will print to a given {@link OutputStream}
   *
   * @param outputStream the output to print
   */
  public FeaturizerEvaluationErrorListener(OutputStream outputStream) {
    super(outputStream);
  }

  @Override
  public void missclassified(FeatureSample reference, FeatureSample prediction) {
    printError(reference.getFeatures(), prediction.getFeatures(), reference,
        prediction, reference.getSentence());
  }

}
