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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.Charset;

import opennlp.tools.cmdline.ArgumentParser;
import opennlp.tools.cmdline.ArgumentParser.OptionalParameter;
import opennlp.tools.cmdline.ArgumentParser.ParameterDescription;
import opennlp.tools.cmdline.CmdLineUtil;
import opennlp.tools.cmdline.StreamFactoryRegistry;
import opennlp.tools.formats.LanguageSampleStreamFactory;
import opennlp.tools.postag.POSSample;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

/**
 * <b>Note:</b> Do not use this class, internal use only!
 */
public class ADExPOSSampleStreamFactory extends
    LanguageSampleStreamFactory<POSSample> {

  interface Parameters {
    @ParameterDescription(valueName = "charsetName", description = "encoding for reading and writing text, if absent the system default is used.")
    Charset getEncoding();

    @ParameterDescription(valueName = "sampleData", description = "data to be used, usually a file name.")
    File getData();

    @ParameterDescription(valueName = "language", description = "language which is being processed.")
    String getLang();

    @ParameterDescription(valueName = "expandME", description = "expand multiword expressions.")
    @OptionalParameter(defaultValue = "false")
    Boolean getExpandME();

    @ParameterDescription(valueName = "includeFeatures", description = "combine POS Tags with word features, like number and gender.")
    @OptionalParameter(defaultValue = "false")
    Boolean getIncludeFeatures();
    
    @ParameterDescription(valueName = "addContext", description = "include additional context")
    @OptionalParameter(defaultValue = "false")
    Boolean getIncludeAdditionalContext();
  }

  public static void registerFactory() {
    StreamFactoryRegistry.registerFactory(POSSample.class, "adex",
        new ADExPOSSampleStreamFactory(Parameters.class));
  }

  protected <P> ADExPOSSampleStreamFactory(Class<P> params) {
    super(params);
  }

  public ObjectStream<POSSample> create(String[] args) {

    Parameters params = ArgumentParser.parse(args, Parameters.class);

    language = params.getLang();

    FileInputStream sampleDataIn = CmdLineUtil.openInFile(params.getData());

    ObjectStream<String> lineStream = null;
    try {
      lineStream = new PlainTextByLineStream(
          new MarkableFileInputStreamFactory(params.getData()), params.getEncoding());
    } catch (IOException e) {
      e.printStackTrace();
    }

    ADExPOSSampleStream sentenceStream = new ADExPOSSampleStream(lineStream,
        params.getExpandME(), params.getIncludeFeatures(), params.getIncludeAdditionalContext());

    return sentenceStream;
  }

}
