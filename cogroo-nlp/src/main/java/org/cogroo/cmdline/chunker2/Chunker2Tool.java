/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.cogroo.cmdline.chunker2;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.cmdline.BasicCmdLineTool;
import opennlp.tools.cmdline.CLI;
import opennlp.tools.cmdline.CmdLineUtil;
import opennlp.tools.cmdline.PerformanceMonitor;
import opennlp.tools.postag.POSSample;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

import org.cogroo.tools.chunker2.ChunkerME;
import org.cogroo.tools.chunker2.ChunkerModel;

public class Chunker2Tool extends BasicCmdLineTool {

  public String getShortDescription() {
    return "learnable chunker";
  }
  
  public String getHelp() {
    return "Usage: " + CLI.CMD + " " + getName() + " model < sentences";
  }

  public void run(String[] args) {
    if (args.length != 1) {
      System.out.println(getHelp());
    } else {
      ChunkerModel model = new ChunkerModelLoader().load(new File(args[0]));

      ChunkerME chunker = new ChunkerME(model, ChunkerME.DEFAULT_BEAM_SIZE);

      InputStreamReader isr = new InputStreamReader(System.in);
      BufferedReader br = new BufferedReader(isr);
      try {
        String fileName = br.readLine();
        File fileIn = new File(fileName);
        ObjectStream<String> lineStream =
                new PlainTextByLineStream(new MarkableFileInputStreamFactory(fileIn), "utf-8");

        PerformanceMonitor perfMon = new PerformanceMonitor(System.err, "sent");
        perfMon.start();
        String line;
        while ((line = lineStream.read()) != null) {

          POSSample posSample;
          try {
            posSample = POSSample.parse(line);
          } catch (InvalidFormatException e) {
            System.err.println("Invalid format:");
            System.err.println(line);
            continue;
          }

          String[] chunks = chunker.chunk(posSample.getSentence(),
                  posSample.getTags());

          System.out.println(new ChunkSample(posSample.getSentence(),
                  posSample.getTags(), chunks).nicePrint());

          perfMon.incrementCounter();
        }
        perfMon.stopAndPrintFinalResult();

      } catch (IOException e) {
        e.printStackTrace();
        CmdLineUtil.handleStdinIoError(e);
      }




    }
  }
}
