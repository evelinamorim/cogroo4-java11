package org.cogroo.tools.checker.checkers;

import java.util.List;
import java.util.Locale;

import org.cogroo.analyzer.Analyzer;
import org.cogroo.analyzer.ComponentFactory;
import org.cogroo.entities.Mistake;
import org.cogroo.text.Document;
import org.cogroo.text.Sentence;
import org.cogroo.text.impl.DocumentImpl;
import org.junit.BeforeClass;
import org.junit.Test;

import static junit.framework.Assert.assertEquals;
import static org.junit.Assert.*;

public class GovernmentCheckerTest {

	private static Analyzer analyzer;

	@BeforeClass
	public static void init() {

		analyzer = ComponentFactory.create(new Locale("pt", "BR")).createPipe();
	}

	private Sentence createSentence(String test) {
		Document document = new DocumentImpl();
		document.setText(test);

		analyzer.analyze(document);

		return document.getSentences().get(0);
	}

	public void testWrong(String test) {
		Sentence sentence = createSentence(test);

		GovernmentChecker checker = new GovernmentChecker();
		List<Mistake> errors = checker.check(sentence);

		assertNotSame(0, errors.size());
	}

	public void testRight(String test) {
		Sentence sentence = createSentence(test);

		GovernmentChecker checker = new GovernmentChecker();
		List<Mistake> errors = checker.check(sentence);

		assertEquals(0, errors.size());
	}

	@Test
	public void tests() {
		testRight("Ele assistiu à peça de teatro do seu amigo");
		testWrong("Ele assistiu a peça de teatro do seu amigo");
		testRight("O médico assistiu o paciente");
		testWrong("O médico assistiu ao paciente");

//		testRight("João simpatizou com aquela garota");
//		testWrong("João simpatizou naquela garota");
		// The problem is the wrong object identification

		testRight("Ele sempre aspirou ao cargo de gerente");
		testWrong("Ele sempre aspirou o cargo de gerente");
		testRight("Ela aspirou todo o pó ontem");
		testWrong("Ela aspirou a todo o pó ontem");
		testRight("Ela aspirou todo o pó da sala");

		testRight("Atendeu à solicitação");
		testRight("Atendeu o telefone");

//		testRight("Ele namorou a garota");
//		testWrong("Ele namorou com a garota");
//		testRight("Ele namorava aquela menina com câncer");
//		testRight("Ele namorou a filha do chefe");
//		testRight("Ele namorava com vigor aquela garota");
		
		testRight("A criança obedeceu ao pai");
		testWrong("A criança obedeceu o pai");
		
//		testRight("Ela usufruiu durante anos os bens alheios");
//		testWrong("Ela usufruiu dos bens alheios");
//		testRight("Ela usufruía o salário do marido");
		
//		testRight("Ele simpatizava com a atendente da loja");
//		testWrong("Ele simpatizava a ela");
		
	}

	@Test
	public void testToBeCorrected() {
		testRight("O médico assistiu de imediato na sala do hospital o doente");
		testRight("Ela aspirou todo o pó da sala");
//		testRight("Ele namorou a filha do chefe");
		
		//The following tests are wrong because of the preposition in the
		//direct object
	}
}
