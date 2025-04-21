const taskForm = document.getElementById('task-form');
const taskInput = document.getElementById('task-input');
const chatOutput = document.getElementById('chat-output');
const taskIdDisplay = document.getElementById('task-id-display');
const statusDisplay = document.getElementById('status-display');
const resultDisplay = document.getElementById('result-display');
const errorDisplay = document.getElementById('error-display');
const urlDisplay = document.getElementById('url-display');
const screenshotDisplay = document.getElementById('screenshot-display');

let currentTaskId = null;
let websocket = null;

// Function to add messages to chat
function addChatMessage(sender, message, type = 'agent') {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type);
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatOutput.appendChild(messageDiv);
    chatOutput.scrollTop = chatOutput.scrollHeight; // Scroll to bottom
}

// Function to update status displays
function updateStatusDisplays(status, result, error) {
    statusDisplay.textContent = status || 'N/A';
    resultDisplay.textContent = result ? JSON.stringify(result, null, 2) : 'N/A';
    errorDisplay.textContent = error || 'N/A';
    errorDisplay.style.display = error ? 'block' : 'none'; // Show/hide error block
}

// --- Event Listeners ---

taskForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const taskDescription = taskInput.value.trim();
    if (!taskDescription) return;

    addChatMessage('You', taskDescription, 'user');
    taskInput.value = ''; // Clear input
    updateStatusDisplays('Starting...', null, null);
    taskIdDisplay.textContent = 'Starting...';
    urlDisplay.textContent = 'N/A';
    screenshotDisplay.src = ''; // Clear screenshot

    // Close existing websocket if any
    if (websocket) {
        websocket.close();
        websocket = null;
    }

    try {
        const response = await fetch('/api/start_task', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ task_description: taskDescription })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        currentTaskId = data.task_id;
        taskIdDisplay.textContent = currentTaskId;
        addChatMessage('System', `Task started with ID: ${currentTaskId}`);
        updateStatusDisplays('Pending', null, null);

        // Establish WebSocket connection
        connectWebSocket(currentTaskId);

    } catch (error) {
        console.error('Error starting task:', error);
        addChatMessage('System', `Error starting task: ${error.message}`, 'error');
        updateStatusDisplays('Error', null, error.message);
        taskIdDisplay.textContent = 'Error';
    }
});

// --- WebSocket Logic ---

function connectWebSocket(taskId) {
    // Construct WebSocket URL (adjust if not localhost/8000)
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/${taskId}`;
    
    addChatMessage('System', `Connecting to WebSocket at ${wsUrl}...`);
    websocket = new WebSocket(wsUrl);

    websocket.onopen = (event) => {
        addChatMessage('System', 'WebSocket connection established.');
        console.log('WebSocket connection established.');
    };

    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data);

            if (data.type === 'status_update') {
                addChatMessage('Agent', `Status: ${data.status || 'N/A'}`);
                updateStatusDisplays(data.status, data.result, data.error);
                
                // Update visualization components
                urlDisplay.textContent = data.current_url || 'N/A'; // Assuming backend sends 'current_url'
                if (data.screenshot_base64) {
                    screenshotDisplay.src = `data:image/png;base64,${data.screenshot_base64}`;
                } else {
                   // Maybe keep previous screenshot or clear?
                   // screenshotDisplay.src = ''; 
                }

                // Handle final states
                if (data.status === 'success' || data.status === 'error') {
                    if (websocket) websocket.close(); // Close WS on final state
                    currentTaskId = null;
                }
            } else if (data.type === 'agent_action') { // Example: Handle specific action messages
                 addChatMessage('Agent Action', `${data.tool_name}(${JSON.stringify(data.tool_args)})`);
                 // Could update URL based on navigate action etc.
                 if(data.tool_name === 'navigate') {
                     urlDisplay.textContent = data.tool_args?.url || 'Navigating...';
                 }
            } else if (data.type === 'ack') {
                 // Ignore acknowledgements for now
            } else {
                 addChatMessage('WebSocket', JSON.stringify(data));
            }
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
            addChatMessage('System', `Error processing message: ${error}`, 'error');
        }
    };

    websocket.onerror = (event) => {
        console.error('WebSocket error:', event);
        addChatMessage('System', 'WebSocket error occurred.', 'error');
        updateStatusDisplays('Error', null, 'WebSocket connection error.');
        currentTaskId = null;
        taskIdDisplay.textContent = 'Error';
    };

    websocket.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        addChatMessage('System', `WebSocket connection closed (${event.code}).`);
        // Only nullify if it wasn't closed intentionally by reaching final state
        // if (currentTaskId) { 
        //     updateStatusDisplays('Closed', null, 'WebSocket connection closed.');
        // }
        websocket = null; 
        // Optionally try to reconnect or prompt user
    };
} 