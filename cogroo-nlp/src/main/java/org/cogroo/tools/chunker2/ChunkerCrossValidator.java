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
package org.cogroo.tools.chunker2;

import java.io.IOException;

import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.eval.CrossValidationPartitioner;
import opennlp.tools.util.eval.FMeasure;
import opennlp.tools.util.model.ModelUtil;

public class ChunkerCrossValidator {

  private final String languageCode;
  private final TrainingParameters params;

  private FMeasure fmeasure = new FMeasure();
  private ChunkerEvaluationMonitor[] listeners;
  private ChunkerFactory chunkerFactory;

  /**
   * @deprecated Use
   *             {@link #ChunkerCrossValidator(String, TrainingParameters, ChunkerFactory, ChunkerEvaluationMonitor...)}
   *             instead.
   * @param languageCode deprecated
   * @param cutoff deprecated
   * @param iterations deprecated
   */
  @Deprecated
  public ChunkerCrossValidator(String languageCode, int cutoff, int iterations) {

    this.languageCode = languageCode;

    //params = ModelUtil.createTrainingParameters(iterations, cutoff);
    params = ModelUtil.createDefaultTrainingParameters();
    listeners = null;
  }

  /**
   * @deprecated Use {@link #ChunkerCrossValidator(String, TrainingParameters, ChunkerFactory, ChunkerEvaluationMonitor...)} instead.
   *
   * @param languageCode code
   * @param params params to training
   * @param listeners monitor
   */
  public ChunkerCrossValidator(String languageCode, TrainingParameters params,
      ChunkerEvaluationMonitor... listeners) {

    this.languageCode = languageCode;
    this.params = params;
    this.listeners = listeners;
  }

  /**
   * @param languageCode code
   * @param params training params
   * @param factory the factory chunker
   * @param listeners monitor
   *
   */

  
  public ChunkerCrossValidator(String languageCode, TrainingParameters params,
      ChunkerFactory factory, ChunkerEvaluationMonitor... listeners) {
    this.chunkerFactory = factory;
    this.languageCode = languageCode;
    this.params = params;
    this.listeners = listeners;
  }

  /**
   * Starts the evaluation.
   * 
   * @param samples
   *          the data to train and test
   * @param nFolds
   *          number of folds
   * 
   * @throws IOException if it is not possible to read the data
   * @throws InvalidFormatException if it in the wrong format
   *
   */
  public void evaluate(ObjectStream<ChunkSample> samples, int nFolds)
      throws IOException, InvalidFormatException, IOException {
    CrossValidationPartitioner<ChunkSample> partitioner = new CrossValidationPartitioner<ChunkSample>(
        samples, nFolds);

    while (partitioner.hasNext()) {

      CrossValidationPartitioner.TrainingSampleStream<ChunkSample> trainingSampleStream = partitioner
          .next();

      ChunkerModel model = ChunkerME.train(languageCode, trainingSampleStream,
          params, chunkerFactory);

      // do testing
      ChunkerEvaluator evaluator = new ChunkerEvaluator(new ChunkerME(model,
          ChunkerME.DEFAULT_BEAM_SIZE), listeners);

      evaluator.evaluate(trainingSampleStream.getTestSampleStream());

      fmeasure.mergeInto(evaluator.getFMeasure());
    }
  }

  public FMeasure getFMeasure() {
    return fmeasure;
  }
}
