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
package org.cogroo.tools.featurizer;

import java.io.IOException;
import java.util.Iterator;

import opennlp.tools.util.AbstractEventStream;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.ml.model.Event;

/**
 * Class for creating an event stream out of data files for training a chunker.
 */
public class FeaturizerEventStream extends opennlp.tools.util.AbstractEventStream {

  private FeaturizerContextGenerator cg;
  private ObjectStream<FeatureSample> data;
  private Event[] events;
  private int ei;

  /**
   * Creates a new event stream based on the specified data stream using the
   * specified context generator.
   * 
   * @param d
   *          The data stream for this event stream.
   * @param cg
   *          The context generator which should be used in the creation of
   *          events for this event stream.
   */
  public FeaturizerEventStream(ObjectStream<FeatureSample> d,
      FeaturizerContextGenerator cg) {
    super(d);
    this.cg = cg;
    data = d;
    ei = 0;
    addNewEvents();
  }

  public Event next() {

    hasNext();

    return events[ei++];
  }

  public boolean hasNext() {
    if (ei == events.length) {
      addNewEvents();
      ei = 0;
    }
    return ei < events.length;
  }

  private void addNewEvents() {

    FeatureSample sample;
    try {
      sample = data.read();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    if (sample != null) {
      events = new Event[sample.getSentence().length];
      String[] toksArray = sample.getSentence();
      String[] tagsArray = sample.getTags();
      String[] predsArray = sample.getFeatures();
      for (int ei = 0, el = events.length; ei < el; ei++) {
        events[ei] = new Event(predsArray[ei], cg.getContext(ei, toksArray,
            tagsArray, predsArray));
      }
    } else {
      events = new Event[0];
    }
  }

  @Override
  protected Iterator<Event> createEvents(Object o) {
    return null;
  }
}
