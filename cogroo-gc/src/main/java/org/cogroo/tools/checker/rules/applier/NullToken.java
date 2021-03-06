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
package org.cogroo.tools.checker.rules.applier;

import org.cogroo.entities.Token;

public class NullToken extends Token {

  /**
   * 
   */
  private static final long serialVersionUID = 1L;

  private static final NullToken instance = new NullToken();
  
  public void setLexeme(String lexeme) {
    // do nothing
  }
  
  private NullToken() {
    
  }
  
  public static Token instance() {
    return instance;
  }

  @Override
  public String toString() {
    return "NULL";
  }
}
