from __future__ import annotations

from typing import List

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from model import get_llm

_SYSTEM_PROMPT = (
    "Anda adalah chatbot pintar yang membantu peternak unggas memahami "
    "dan mengatasi penyakit unggas berdasarkan referensi buku "
    "manual dokter hewan.\n\n"
    "Tugas anda adalah menjawab pertanyaan dari user.\n\n"
    "Gunakan history conversation jika ingin melihat pesan sebelumnya."
)

def _build_chain(llm: BaseChatModel) -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["input", "history"],
        template=(
            _SYSTEM_PROMPT
            + "\n\nHistory:\n{history}\n\nPertanyaan:\n{input}\n\n"
            "Jawablah pertanyaan berikut dengan:\n"
            "- Jangan gunakan tanda bintang (*)\n"
            "- Singkat dan jelas\n"
            "- Bahasa Indonesia yang mudah dipahami orang awam\n"
            "- Gaya yang ramah dan membumi\n"
            "- Penjelasan singkat jika ada istilah teknis\n"
            "- Jika pertanyaan hanya basa-basi, bertanyalah balik terlebih dulu\n\n"
            "Jawaban:"
        ),
    )
    return LLMChain(llm=llm, prompt=prompt)


class ChatService:
    """Pembungkus LLMChain agar mudah dipakai di layer Flask."""

    MAX_HISTORY: int = 10

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._chain = _build_chain(llm or get_llm())

    def reply(self, user_msg: str, history: List[str]) -> str:
        """Kembalikan balasan LLM berdasar `user_msg` & `history`."""
        recent = "\n".join(history[-self.MAX_HISTORY :])
        return self._chain.run({"input": user_msg, "history": recent})
