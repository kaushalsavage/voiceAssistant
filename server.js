/**
 * @brief Voice assistant. Server side NodeJS.
 * @author Yurii Mykhailov
 * @copyright GPLv3
 */

// include libs
const express = require('express');
const fs = require('fs');
const path = require('path');
const { HfInference } = require('@huggingface/inference');
const config = require('./config');

// Init express
const app = express();
const port = 3000;

// Path to files
const recordFile = path.resolve("./resources/recording.wav");
const voicedFile = path.resolve("./resources/voicedby.wav");

// API Key and settings
const hfToken = config.hfToken;
let shouldDownloadFile = false;
const maxTokens = 30;

// Init HuggingFace
const hf = new HfInference(hfToken);

// Middleware for data processing in a "multipart/form-data" format 
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Handler for loading an audio file
app.post('/uploadAudio', (req, res) => {
    shouldDownloadFile = false;
    
    // Ensure resources directory exists
    if (!fs.existsSync('./resources')) {
        fs.mkdirSync('./resources');
    }

    let dataReceived = false;
    const recordingFile = fs.createWriteStream(recordFile, { 
        encoding: 'binary',
        flags: 'w' // Overwrite existing file
    });
    
    recordingFile.on('error', (err) => {
        console.error('Error writing file:', err);
        res.status(500).send('Error saving audio file');
    });

    req.on('error', (err) => {
        console.error('Error receiving data:', err);
        res.status(500).send('Error receiving audio data');
    });

    req.on('data', (data) => {
        dataReceived = true;
        console.log('Receiving audio chunk, size:', data.length);
        recordingFile.write(data);
    });

    req.on('end', async () => {
        recordingFile.end();
        
        if (!dataReceived) {
            console.error('No audio data received');
            return res.status(400).send('No audio data received');
        }

        // Wait for file to be fully written
        await new Promise(resolve => setTimeout(resolve, 100));

        try {
            const transcription = await speechToTextAPI();
            if (!transcription) {
                return res.status(500).send('Speech recognition failed');
            }
            console.log('Transcription successful:', transcription);
            res.status(200).send(transcription);
            await callGPT(transcription);
        } catch (error) {
            console.error('Audio processing error:', error);
            res.status(500).send('Error processing audio');
        }
    });
});

// Handler for checking the value of a variable
app.get('/checkVariable', (req, res) => {
	res.json({ ready: shouldDownloadFile });
});

// File upload handler
app.get('/broadcastAudio', (req, res) => {

	fs.stat(voicedFile, (err, stats) => {
		if (err) {
			console.error('File not found');
			res.sendStatus(404);
			return;
		}

		res.writeHead(200, {
			'Content-Type': 'audio/wav',
			'Content-Length': stats.size
		});

		const readStream = fs.createReadStream(voicedFile);
		readStream.pipe(res);

		readStream.on('end', () => {
			//console.log('File has been sent successfully');
		});

		readStream.on('error', (err) => {
			console.error('Error reading file', err);
			res.sendStatus(500);
		});
	});
});

// Starting the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});

// Remove these duplicate lines:
// const { HfInference } = require('@huggingface/inference');
// const hf = new HfInference(config.hfToken);

// Keep existing functions
async function speechToTextAPI() {
    try {
        const fileStats = await fs.promises.stat(recordFile);
        if (fileStats.size === 0) {
            throw new Error('Audio file is empty');
        }

        const audioData = await fs.promises.readFile(recordFile);
        const transcription = await hf.automaticSpeechRecognition({
            data: audioData,
            model: "openai/whisper-base",
            parameters: {
                return_timestamps: false,
                language: "en"
            }
        });
        console.log('YOU:', transcription.text);
        return transcription.text;
    } catch (error) {
        console.error('Error in speechToTextAPI:', error);
        return null;
    }
}

async function callGPT(text) {
    try {
        if (!text) return null;
        const response = await hf.textGeneration({
            inputs: `Answer briefly: ${text}`,
            model: "gpt2",
            parameters: {
                max_new_tokens: 50,
                temperature: 0.5,
                return_full_text: false,
                repetition_penalty: 1.2
            }
        });

        const aiResponse = response.generated_text;
        console.log('AI:', aiResponse);
        
        await GptResponsetoSpeech(aiResponse);
        return aiResponse;
    } catch (error) {
        console.error('Error calling AI:', error);
        return null;
    }
}

async function GptResponsetoSpeech(text) {
    try {
        if (!fs.existsSync('./resources')) {
            fs.mkdirSync('./resources');
        }

        const response = await hf.textToSpeech({
            model: 'espnet/kan-bayashi_ljspeech_vits',
            inputs: text.trim()
        });

        const buffer = Buffer.from(await response.arrayBuffer());
        await fs.promises.writeFile(voicedFile, buffer);
        shouldDownloadFile = true;
        console.log('Audio generated successfully');
    } catch (error) {
        console.error("Error in text-to-speech:", error);
        shouldDownloadFile = false;
    }
}

// Add this with your other route handlers
// Add at the top with other requires
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Update Gemini initialization
const genAI = new GoogleGenerativeAI(config.geminiApiKey);
const model = genAI.getGenerativeModel({ 
    model: "gemini-2.0-flash",  // Updated model name
    apiVersion: "v1beta"  // Specify API version
});

// Update the sendText endpoint
app.post('/sendText', async (req, res) => {
    try {
        const userText = req.body.text;
        if (!userText) {
            return res.status(400).send('Text input required');
        }
        
        console.log('User Input:', userText);
        
        const result = await model.generateContent({
            contents: [{
                parts: [{ text: userText }]
            }]
        });
        
        const response = await result.response;
        const aiResponse = response.text();
        
        console.log('AI Response:', aiResponse);
        await GptResponsetoSpeech(aiResponse);
        
        res.status(200).json({ 
            response: aiResponse,
            audioReady: shouldDownloadFile 
        });
    } catch (error) {
        console.error('Error processing text:', error);
        res.status(500).send('Error processing request: ' + error.message);
    }
});
