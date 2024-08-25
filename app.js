import express from 'express';
import http from 'http';
import { fileURLToPath } from 'url';
import path, { dirname } from 'path';
import * as dotenv from 'dotenv';
import { Fireworks } from '@langchain/community/llms/fireworks';
import { FireworksEmbeddings } from '@langchain/community/embeddings/fireworks';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { loadQAStuffChain } from 'langchain/chains';
import fs from 'fs';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const app = express();
const port = 3000;

/* Create HTTP server */
http.createServer(app).listen(process.env.PORT);

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

// Endpoint to check current status
app.get('/api/health', async (req, res) => {
    res.json({
        success: true,
        message: 'Server is healthy',
    });
});

// Endpoint to ask questions
app.get('/ask', async (req, res) => {
    try {
        const llmA = new Fireworks({ temperature: 0.9 });
        const chainA = loadQAStuffChain(llmA);
        const pdfPath = "Arrays.pdf"; // Path to the PDF file
        const loader = new PDFLoader(pdfPath);
        const docs = await loader.load();

        const embeddings = new FireworksEmbeddings();
        const vectorStore = await HNSWLib.fromDocuments(docs, embeddings);

        const directory = process.env.DIR;
        if (!fs.existsSync(directory)) {
            fs.mkdirSync(directory);
        }
        
        await vectorStore.save(directory);

        const question = "what are arrays used for?"; // Question to ask
        const result = await vectorStore.similaritySearch(question, 1);
        const resA = await chainA._call({
            input_documents: result,
            question,
        });
        
        //res.json({ result: resA });

        // Clean the text in the result
        let cleanText = resA.text;
        cleanText = cleanText.replace(/\n/g, ''); // Remove newlines
        cleanText = cleanText.replace(/\t/g, ''); // Remove tabs
        cleanText = cleanText.replace(/\r/g, ''); // Remove carriage returns
        resA.text = cleanText;

        // Pretty print the JSON response
        const prettyResponse = JSON.stringify({ result: resA }, null, 2);

        res.setHeader('Content-Type', 'application/json');
        res.send(prettyResponse);

    } catch (error) {
        console.log(error);
        res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
    }
});
