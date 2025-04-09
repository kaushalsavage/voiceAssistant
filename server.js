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
                    parts: [{ text }]
                }]
            },
            {
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        );
        return response.data.candidates[0].content.parts[0].text;
    } catch (error) {
        console.error('Gemini API Error:', error.response?.data || error.message);
        throw error;
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
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Routes
app.post('/uploadAudio', (req, res) => {
    shouldDownloadFile = false;
    
    if (!fs.existsSync('./resources')) {
        fs.mkdirSync('./resources');
    }

    let dataReceived = false;
    const recordingFile = fs.createWriteStream(recordFile, { 
        encoding: 'binary',
        flags: 'w'
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

        await new Promise(resolve => setTimeout(resolve, 100));

        try {
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
            console.error('Processing error:', error);
            res.status(500).send('Error processing request');
        }
    });
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
