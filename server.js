const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const multer = require('multer');

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function(req, file, cb) {
        // Create uploads directory if it doesn't exist
        if (!fs.existsSync('uploads')) {
            fs.mkdirSync('uploads');
        }
        cb(null, 'uploads/');
    },
    filename: function(req, file, cb) {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ storage: storage });

// Serve static files
app.use(express.static(path.join(__dirname)));
app.use('/uploads', express.static('uploads'));

// Add a test route
app.get('/', (req, res) => {
    console.log('Received request for /');
    res.send('Server is running!');
});

app.get('/chat.html', (req, res) => {
    console.log('Received request for chat.html');
    res.sendFile(path.join(__dirname, 'chat.html'));
});

// Store connected users
const users = new Map();

async function processImage(imagePath) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['image_processor.py', imagePath]);
        let result = '';

        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(`Process exited with code ${code}`);
            } else {
                try {
                    resolve(JSON.parse(result));
                } catch (e) {
                    reject('Failed to parse Python output');
                }
            }
        });
    });
}

io.on('connection', (socket) => {
    console.log('A user connected');

    socket.on('user-join', (username) => {
        users.set(socket.id, username);
        io.emit('user-joined', username);
    });

    socket.on('chat-message', async (data) => {
        try {
            if (data.fileData && data.fileType) {
                // Handle file upload
                const base64Data = data.fileData.split(';base64,').pop();
                const fileExt = data.fileType.split('/')[1];
                const fileName = `uploads/${Date.now()}.${fileExt}`;

                // Save file
                fs.writeFile(fileName, base64Data, {encoding: 'base64'}, async (err) => {
                    if (err) {
                        console.error('Error saving file:', err);
                        return;
                    }

                    let mlResult = null;
                    if (data.fileType.startsWith('image/')) {
                        try {
                            mlResult = await processImage(fileName);
                        } catch (error) {
                            console.error('ML processing error:', error);
                        }
                    }

                    // Emit message with file
                    io.emit('message', {
                        username: data.username || 'Anonymous',
                        message: data.message,
                        fileData: data.fileData,
                        fileType: data.fileType,
                        mlPrediction: mlResult
                    });
                });
            } else {
                // Regular text message
                io.emit('message', {
                    username: data.username || 'Anonymous',
                    message: data.message
                });
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    });

    socket.on('disconnect', () => {
        const username = users.get(socket.id);
        users.delete(socket.id);
        io.emit('user-left', username);
        console.log('User disconnected');
    });
});

const PORT = process.env.PORT || 3000;
http.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Open http://localhost:${PORT}/chat.html in your browser`);
}); 