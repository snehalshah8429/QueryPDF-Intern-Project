import { Fireworks } from "@langchain/community/llms/fireworks";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

//import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

import { FireworksEmbeddings } from "@langchain/community/embeddings/fireworks";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import * as dotenv from 'dotenv'
dotenv.config()

export const injest_docs = async() => {
    const loader = new PDFLoader("Arrays.pdf"); //can be changed to any pdf file of my choice
    const docs = await loader.load();
    console.log('docs loaded')

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      const docOutput = await textSplitter.splitDocuments(docs)
      let vectorStore = await HNSWLib.fromDocuments(
        docOutput,
        new FireworksEmbeddings(),
      )

      console.log('saving...')

      const directory = "/Users/snehalkumar8429/Desktop/chatwithpdf/";
      await vectorStore.save (directory);
      console.log("Saved");
}

injest_docs();