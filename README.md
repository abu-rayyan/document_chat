This is the backend code of a chatbot. 
This chatbot has 4 modules. 
1st module does not include any RAG. It simply take user query along with history and reply to user based on LLM model.
2nd module also does not use RAGs and will work on documents on the go. User will be able to upload any document and will get replies from that document. It wont use RAG. It will only work for short documents.
3rd module will use RAG with history. Features like reranking and subquery optimization will be optional. 
4th module will use Agentic AI to respond to user queries. 
