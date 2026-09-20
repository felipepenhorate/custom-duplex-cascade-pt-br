"""Banco Penha domain agent: a tiny stateful controller that watches the
user's words in REAL TIME (before the LLM answers) and can take the floor
itself — a SYSTEM barge-in, the duplex counterpart of the user barge-in.

Flow (SPEC: the user's scenario):
  idle        -- "quero um cartão de crédito" --> ask_cpf (system asks
                 for the CPF), state -> await_cpf
  await_cpf   -- the user dictates the CPF; the agent watches every word:
                 * a LETTER instead of a number --> cpf_letters (system
                   barge-in: stops the user, only numbers are allowed)
                 * 11 digits collected          --> cpf_ok (system confirms)
  cpf_done    -- back to normal conversation

The agent returns an ACTION per utterance; the bridge executes it (reset
the user's turn, speak the message, show it in the chat).
"""

from __future__ import annotations

import re

CREDIT_CARD_KEYWORDS = [
    "cartão de crédito",
    "cartao de credito",
    "cartão de credito",
    "cartao de crédito",
    "cartão",
    "cartao",
    "crédito",
    "credito",
]

# spoken digit words must NOT be treated as "letters"
NUMBER_WORDS = {
    "zero": "0",
    "um": "1",
    "dois": "2",
    "três": "3",
    "tres": "3",
    "quatro": "4",
    "cinco": "5",
    "seis": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
}

MSG_ASK_CPF = (
    "Claro! Para solicitar o seu cartão de crédito, vou precisar do seu "
    "CPF. Por favor, diga os onze números."
)
MSG_CPF_LETTERS = (
    "Ops! No CPF são permitidos apenas números. Pode repetir, por favor?"
)
MSG_CPF_OK = (
    "Perfeito, recebi seu CPF. Já estou providenciando o seu cartão de "
    "crédito do Banco Penha."
)

# control instructions sent to the BIG model (Banco Penha persona) so it
# generates the bank's response naturally; the canned messages above are
# only the fallback if generation fails
BANK_INSTRUCTIONS = {
    "ask_cpf": (
        "O cliente quer solicitar um cartão de crédito. Informe que você "
        "precisa do CPF dele para dar continuidade e peça que ele diga os "
        "onze números."
    ),
    "cpf_letters": (
        "O cliente estava dizendo o CPF e falou uma letra em vez de um "
        "número. Avise educadamente que no CPF são permitidos apenas "
        "números e peça para ele repetir."
    ),
    "cpf_ok": (
        "O cliente terminou de dizer o CPF. Confirme que você recebeu os "
        "dados e que o cartão de crédito do Banco Penha está sendo "
        "providenciado."
    ),
}


class BankAgent:
    def __init__(self) -> None:
        self.state = "idle"  # idle | await_cpf | cpf_done
        self.digits = ""
        self.fired = False  # an action fired for the current utterance

    def reset_utterance(self) -> None:
        self.fired = False

    def on_words(self, words: str):
        """Watch the words of the current utterance. Returns an action
        (str, ...) tuple or None."""
        w = words.strip().lower()
        if not w or self.fired:
            return None

        if self.state == "idle":
            if any(kw in w for kw in CREDIT_CARD_KEYWORDS):
                self.state = "await_cpf"
                self.fired = True
                return ("ask_cpf",)
            return None

        if self.state == "await_cpf":
            # letters (excluding spoken digit words) = invalid CPF digit
            letter_words = [x for x in re.findall(r"[a-zà-ú]+", w)
                            if x not in NUMBER_WORDS]
            if letter_words:
                self.fired = True
                self.digits = ""  # the user must start the CPF over
                return ("cpf_letters", letter_words[0])
            # accumulate digits from numerals and spoken digit words
            for ch in w:
                if ch.isdigit():
                    self.digits += ch
            for x in re.findall(r"[a-zà-ú]+", w):
                self.digits += NUMBER_WORDS.get(x, "")
            if len(self.digits) >= 11:
                self.fired = True
                cpf = self.digits[:11]
                self.state = "cpf_done"
                return ("cpf_ok", cpf)
            return None
        return None

    @staticmethod
    def message(action: tuple) -> str:
        kind = action[0]
        if kind == "ask_cpf":
            return MSG_ASK_CPF
        if kind == "cpf_letters":
            return MSG_CPF_LETTERS
        if kind == "cpf_ok":
            return MSG_CPF_OK
        return ""


if __name__ == "__main__":
    # quick self-test (reset_utterance() = a new utterance, like the bridge)
    a = BankAgent()
    assert a.on_words("quero um cartão de crédito") == ("ask_cpf",)
    a.reset_utterance()
    assert a.on_words("um dois três") is None
    a.reset_utterance()
    assert a.on_words("quatro cinco seis") is None
    a.reset_utterance()
    act = a.on_words("sete oito A")
    print("letter action:", act)
    assert act[0] == "cpf_letters"
    a.reset_utterance()
    assert a.on_words("um dois três quatro cinco seis") is None
    a.reset_utterance()
    act = a.on_words("sete oito nove zero um dois")
    print("cpf action:", act)
    assert act[0] == "cpf_ok"
    assert act[1] == "12345678901"
    print("BankAgent OK")