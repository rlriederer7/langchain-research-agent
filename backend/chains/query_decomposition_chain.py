import asyncio
from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


class SubQueries(BaseModel):
    sub_questions: List[str] = Field(
        description="List of simpler sub-questions that together answer the main question"
    )
    reasoning: str = Field(
        description="Brief explanation of the decomposition strategy"
    )


class QueryDecompositionChain:
    def __init__(
            self,
            llm,
            research_agent
    ):
        self.llm = llm
        self.research_agent = research_agent

        self.decomposition_chain = self.create_decomposition_chain()
        self.synthesis_chain = self.create_synthesis_chain()

    def create_decomposition_chain(self):
        parser = PydanticOutputParser(pydantic_object=SubQueries)

        prompt = ChatPromptTemplate.from_template(
            """You are a research assistant that breaks down complex questions into simpler sub-questions.
    
            Given a complex research question, decompose it into 2-5 simpler sub-questions that:
            1. Can be answered independently
            2. Together provide enough information to answer the original question
            3. Are specific and focused
            4. Progress logically from foundational to more complex
    
            Complex Question: {question}
    
            {format_instructions}
    
            Be strategic: sometimes you need background info first, sometimes you need to compare multiple aspects."""
        )

        return (
            {
                "question": RunnablePassthrough(),
                "format_instructions": lambda _: parser.get_format_instructions()
            }
            | prompt
            | self.llm
            | parser
        )

    def create_synthesis_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """You are synthesizing research findings into a comprehensive answer.
    
            Original Question: {original_question}
    
            Sub-questions and their answers:
            {sub_answers}
    
            Task: Provide a well-structured, comprehensive answer to the original question by:
            1. Integrating information from all sub-answers
            2. Resolving any contradictions
            3. Highlighting key insights
            4. Noting any gaps or limitations
    
            Synthesized Answer:"""
        )

        return prompt | self.llm

    async def answer_with_agent(
            self,
            sub_question: str
    ) -> Dict[str, Any]:
        result = await self.research_agent.research(sub_question)

        return {
            "question": sub_question,
            "answer": result["output"]
        }

    async def arun(
            self,
            question: str
    ) -> Dict[str, Any]:
        decomposition = await self.decomposition_chain.ainvoke(question)

        tasks = [
            self.answer_with_agent(sq) for sq in decomposition.sub_questions
        ]
        sub_answers = await asyncio.gather(*tasks)

        sub_answers_formatted = "\n\n".join([
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in sub_answers
        ])

        final_answer = await self.synthesis_chain.ainvoke({
            "original_question": question,
            "sub_answers": sub_answers_formatted
        })

        return {
            "original_question": question,
            "decomposition": {
                "sub_questions": decomposition.sub_questions,
                "reasoning": decomposition.reasoning
            },
            "sub_answers": sub_answers,
            "final_answer": final_answer.content
        }
