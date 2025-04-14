require('dotenv').config();
const express = require('express');
const fs = require('fs');
const path = require('path');
const { HfInference } = require('@huggingface/inference');
const axios = require('axios');

// Init express
const app = express();
const port = process.env.PORT || 3000;

// Path to files
const recordFile = path.resolve("./resources/recording.wav");
const voicedFile = path.resolve("./resources/voicedby.wav");

// Initialize APIs
const hf = new HfInference(process.env.HF_TOKEN);
let shouldDownloadFile = false;

// Audio Processing Functions
async function generateGeminiResponse(text) {
    try {
        const response = await axios.post(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
            {
                contents: [{
                    parts: [{ 
                        text: `Please provide a brief and concise response (maximum 50 words) to: ${text}`
                    }]
                }],
                generationConfig: {
                    maxOutputTokens: 100,
                    temperature: 0.7
                }
            },
            {
                headers: { 'Content-Type': 'application/json' }
            }
        );
        return response.data.candidates[0].content.parts[0].text;
    } catch (error) {
        console.error('Gemini API Error:', error.response?.data || error.message);
        return "I apologize, but I couldn't process that request.";
    }
}

async function speechToTextAPI() {
    try {
        const fileStats = await fs.promises.stat(recordFile);
        if (fileStats.size === 0) throw new Error('Audio file is empty');

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

// Middleware
app.use((req, res, next) => {
    const contentType = req.headers['content-type'] || '';
    
    if (req.path === '/uploadAudio' || contentType.includes('audio') || contentType.includes('application/octet-stream')) {
        const data = [];
        req.on('data', chunk => data.push(chunk));
        req.on('end', () => {
            req.rawBody = Buffer.concat(data);
            next();
        });
        req.on('error', (err) => {
            console.error('Error receiving data:', err);
            res.status(500).send('Error receiving audio data');
        });
    } else {
        express.json({ limit: '50mb' })(req, res, next);
    }
});

app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Update the uploadAudio route
app.post('/uploadAudio', async (req, res) => {
    shouldDownloadFile = false;
    
    if (!fs.existsSync('./resources')) {
        fs.mkdirSync('./resources');
    }

    try {
        if (!req.rawBody) {
            return res.status(400).send('No audio data received');
        }

        await fs.promises.writeFile(recordFile, req.rawBody);
        console.log('Audio file saved, size:', req.rawBody.length);

        const transcription = await speechToTextAPI();
        if (!transcription) {
            return res.status(500).send('Speech recognition failed');
        }
        
        console.log('Transcription successful:', transcription);
        
        const aiResponse = await generateGeminiResponse(transcription);
        console.log('AI Response:', aiResponse);
        
        await GptResponsetoSpeech(aiResponse);
        res.status(200).send(transcription);
    } catch (error) {
        console.error('Error processing audio:', error);
        res.status(500).send('Error processing audio');
    }
});

app.get('/checkVariable', (req, res) => {
    res.json({ ready: shouldDownloadFile });
});

app.get('/broadcastAudio', (req, res) => {
    fs.stat(voicedFile, (err, stats) => {
        if (err) {
            console.error('File not found');
            return res.sendStatus(404);
        }

        res.writeHead(200, {
            'Content-Type': 'audio/wav',
            'Content-Length': stats.size
        });

        const readStream = fs.createReadStream(voicedFile);
        readStream.pipe(res);

        readStream.on('error', (err) => {
            console.error('Error reading file', err);
            res.sendStatus(500);
        });
    });
});

app.post('/sendText', async (req, res) => {
    try {
        const userText = req.body.text;
        if (!userText) {
            return res.status(400).send('Text input required');
        }
        
        console.log('User Input:', userText);
        
        const aiResponse = await generateGeminiResponse(userText);
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

// Start server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});
