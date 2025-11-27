const msgWindow = document.getElementById('message-window');
const form = document.getElementById('input-form');
const input = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const clearButton = document.getElementById('clear-button');
const helpButton = document.getElementById('help-button');
const openChatBtn = document.getElementById('open-chat-btn');
const chatPopup = document.getElementById('chat-popup');
const backdrop = document.getElementById('backdrop');
const pageContent = document.getElementById('page-content');
const feedbackModal = document.getElementById('feedback-modal');
const feedbackBackdrop = document.getElementById('feedback-backdrop');
const feedbackForm = document.getElementById('feedback-form');
const closeFeedbackBtn = document.getElementById('close-feedback-btn');
const feedbackTextarea = document.getElementById('feedback-textarea');
let currentFeedbackContext = {};

const initialBotMessage = '你好！我是慈濟大學獎助學金問答助理，請問有什麼可以幫助您的嗎？';
let chatHistory = [];

// --- Event Listeners ---
form.addEventListener('submit', handleUserSubmit);
clearButton.addEventListener('click', handleClearChat);
helpButton.addEventListener('click', handleGetHelp);
input.addEventListener('input', () => {
    sendButton.disabled = input.value.trim() === '';
});
openChatBtn.addEventListener('click', openChat);
backdrop.addEventListener('click', closeChat);
feedbackForm.addEventListener('submit', handleFeedbackSubmit);
closeFeedbackBtn.addEventListener('click', closeFeedbackModal);
feedbackBackdrop.addEventListener('click', closeFeedbackModal);

// --- Initial State ---
sendButton.disabled = true;
handleClearChat(); // Show examples on initial load

// --- Chat Popup Functions ---
function openChat() {
    chatPopup.style.display = 'block';
    backdrop.style.display = 'block';
    pageContent.classList.add('blurred');
    input.focus();
}

function closeChat() {
    chatPopup.style.display = 'none';
    backdrop.style.display = 'none';
    pageContent.classList.remove('blurred');
}

// --- Event Handlers ---
async function handleUserSubmit(e) {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    const exampleContainer = msgWindow.querySelector('.example-questions-container');
    if (exampleContainer) {
        exampleContainer.remove();
    }

    addMessage(query, 'user');
    chatHistory.push({ role: 'user', content: query });
    input.value = '';
    sendButton.disabled = true;

    const thinkingMessageId = `bot-${Date.now()}`;
    addMessage('<span class="thinking">思考中...</span>', 'bot', thinkingMessageId);

    let fullAnswer = '';
    let isFirstChunk = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                history: chatHistory.slice(0, -1)
            })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                chatHistory.push({ role: 'assistant', content: fullAnswer });
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep the last, possibly incomplete, line

            for (const line of lines) {
                if (line.startsWith('event: end_stream')) {
                    const dataLine = line.substring(line.indexOf('data: ') + 6);
                    const eventData = JSON.parse(dataLine);
                    const finalData = eventData.data;
                    appendBotMessageAddons(thinkingMessageId, finalData.contexts, finalData.log_id);
                } else if (line.startsWith('data:')) {
                    const dataLine = line.substring(6);
                    const eventData = JSON.parse(dataLine);
                    const chunk = eventData.data;

                    if (isFirstChunk) {
                        const messageEl = document.getElementById(thinkingMessageId)?.querySelector('.bot-message');
                        if (messageEl) messageEl.innerHTML = ''; // Clear "Thinking..."
                        isFirstChunk = false;
                    }
                    
                    fullAnswer += chunk;
                    const messageEl = document.getElementById(thinkingMessageId)?.querySelector('.bot-message');
                    if (messageEl) {
                        messageEl.innerHTML = marked.parse(fullAnswer);
                        msgWindow.scrollTop = msgWindow.scrollHeight;
                    }
                } else if (line.startsWith('event: error')) {
                    throw new Error('Server-side error during streaming.');
                }
            }
        }

    } catch (error) {
        console.error('API Error:', error);
        const errorMsg = '抱歉，連線時發生錯誤，請稍後再試。';
        const messageContainer = document.getElementById(thinkingMessageId);
        const messageEl = messageContainer?.querySelector('.bot-message');
        if (messageEl) {
            messageEl.innerHTML = errorMsg;
        } else {
             updateBotMessage(thinkingMessageId, errorMsg); // Fallback
        }
        chatHistory.push({ role: 'assistant', content: errorMsg });
    } finally {
        input.focus();
    }
}

function handleClearChat() {
    msgWindow.innerHTML = '';
    addMessage(initialBotMessage, 'bot');
    chatHistory = [{'role': 'assistant', 'content': initialBotMessage}];
    
    if (!msgWindow.querySelector('.example-questions-container')) {
        const exampleContainer = document.createElement('div');
        exampleContainer.className = 'example-questions-container';
        exampleContainer.innerHTML = `
            <div class="example-question" onclick="askQuestion('提供給五專生原住民的獎助學金有哪些?')">提供給五專生原住民的獎助學金有哪些?</div>
            <div class="example-question" onclick="askQuestion('校內的工讀甚麼時候開放申請?')">校內的工讀甚麼時候開放申請?</div>
            <div class="example-question" onclick="askQuestion('家庭意外補助')">家庭意外補助</div>
            <div class="example-question" onclick="askQuestion('低收入可以申請甚麼?')">低收入可以申請甚麼?</div>
            <div class="example-question" onclick="askQuestion('大三下要到海外交流和志工服務, 學校有提供甚麼補助嗎?')">大三下要到海外交流和志工服務, 學校有提供甚麼補助嗎?</div>
        `;
        msgWindow.appendChild(exampleContainer);
    }
}

function handleGetHelp() {
    alert('聯絡資訊\n\n電話: (03) 856-5301 ext.00000\n郵箱: example@gms.tcu.edu.tw');
}

function askQuestion(question) {
    input.value = question;
    sendButton.disabled = false;
    form.dispatchEvent(new Event('submit'));
}

// --- UI Helper Functions ---
function addMessage(text, sender, id = null) {
    let messageWrapper;

    if (sender === 'user') {
        messageWrapper = document.createElement('div');
        messageWrapper.classList.add('message', 'user-message');
        
        const tempDiv = document.createElement('div');
        tempDiv.textContent = text;
        messageWrapper.innerHTML = tempDiv.innerHTML;
    } else { // sender === 'bot'
        messageWrapper = document.createElement('div');
        messageWrapper.classList.add('bot-message-container');
        if (id) messageWrapper.id = id;

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'bot-message-content';

        const avatar = document.createElement('img');
        avatar.src = 'school_logo.png';
        avatar.alt = 'School Logo';
        avatar.className = 'avatar';

        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message', 'bot-message');
        messageBubble.innerHTML = text; 

        contentWrapper.appendChild(avatar);
        contentWrapper.appendChild(messageBubble);
        messageWrapper.appendChild(contentWrapper);
    }

    msgWindow.appendChild(messageWrapper);
    msgWindow.scrollTop = msgWindow.scrollHeight;
}

function appendBotMessageAddons(id, contexts = [], log_id = null) {
    const messageContainer = document.getElementById(id);
    if (!messageContainer) return;

    const messageEl = messageContainer.querySelector('.bot-message');
    if (!messageEl) return;

    if (contexts && contexts.length > 0) {
        const contextsDiv = document.createElement('div');
        contextsDiv.className = 'contexts';
        let contextsHTML = '<h4>參考資料：</h4>';
        contexts.forEach(ctx => {
            const fileName = ctx.source_file || '未知來源';
            const url = ctx.source_url || '#';
            contextsHTML += `<div class="context-item"><a href="${url}" target="_blank" rel="noopener noreferrer">&#10148; ${fileName}</a></div>`;
        });
        contextsDiv.innerHTML = contextsHTML;
        messageEl.appendChild(contextsDiv);
    }

    if (log_id) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-buttons';
        feedbackDiv.dataset.logId = log_id;
        feedbackDiv.innerHTML = `
            <button class="feedback-btn like-btn" title="滿意"><i class="fa-regular fa-thumbs-up fa-lg" style="color: #adb1b9;"></i></button>
            <button class="feedback-btn dislike-btn" title="不滿意"><i class="fa-regular fa-thumbs-down fa-lg" style="color: #adb1b9;"></i></button>
        `;
        messageContainer.appendChild(feedbackDiv);
        feedbackDiv.addEventListener('click', handleFeedbackClick);
    }
    
    msgWindow.scrollTop = msgWindow.scrollHeight;
}

async function handleFeedbackClick(e) {
    const target = e.target.closest('.feedback-btn');
    if (!target) return;

    const container = target.parentElement;
    const logId = container.dataset.logId;
    if (!logId) return;

    const likeBtn = container.querySelector('.like-btn');
    const dislikeBtn = container.querySelector('.dislike-btn');
    
    const isDislike = target.classList.contains('dislike-btn');
    const isSelected = target.classList.contains('selected');

    // If the clicked button is already selected, unselect it and clear feedback
    if (isSelected) {
        target.classList.remove('selected');
        await sendFeedback(logId, null, null); // Clear both type and text
    } else {
        // Unselect the other button if it's selected
        if (isDislike) {
            likeBtn.classList.remove('selected');
        } else {
            dislikeBtn.classList.remove('selected');
        }
        
        // Select the clicked one
        target.classList.add('selected');

        // If dislike is chosen, save it immediately then open modal
        if (isDislike) {
            await sendFeedback(logId, 'dislike');
            openFeedbackModal(logId);
        } else {
            await sendFeedback(logId, 'like');
        }
    }
}

function openFeedbackModal(logId) {
    currentFeedbackContext = { logId };
    feedbackModal.style.display = 'block';
    feedbackBackdrop.style.display = 'block';
    feedbackTextarea.value = '';
    feedbackTextarea.focus();
}

function closeFeedbackModal() {
    feedbackModal.style.display = 'none';
    feedbackBackdrop.style.display = 'none';
}

async function handleFeedbackSubmit(e) {
    e.preventDefault();
    const feedbackText = feedbackTextarea.value.trim();
    const { logId } = currentFeedbackContext;

    if (!logId) return;

    // Even if the text is empty, we proceed to log the 'dislike'
    await sendFeedback(logId, 'dislike', feedbackText);
    
    closeFeedbackModal();
}

async function sendFeedback(logId, feedbackType, feedbackText = null) {
    console.log('Sending feedback:', { log_id: logId, feedback_type: feedbackType, feedback_text: feedbackText });
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                log_id: parseInt(logId),
                feedback_type: feedbackType,
                feedback_text: feedbackText
            })
        });
        if (!response.ok) throw new Error('Failed to submit feedback');
        console.log('Feedback submitted successfully.');
    } catch (error) {
        console.error('Feedback submission error:', error);
        // Optionally inform the user
        // alert('抱歉，提交回饋時發生錯誤。');
    }
}