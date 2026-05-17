import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e4b-opus-Q5_K_M:1")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local("./db", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model=MODEL, temperature=0.4)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("AI lokal siap. Kirim pertanyaan jaringan kamu.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.message.text
    prompt = f"Jawab dalam bahasa Indonesia, teknis, jelas, dan singkat. Pertanyaan: {q}"
    try:
        ans = qa.invoke({"query": prompt})
        text = ans["result"] if isinstance(ans, dict) and "result" in ans else str(ans)
        await update.message.reply_text(text[:4000])
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
