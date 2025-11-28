// --- DOM Elements ---
const msgWindow = document.getElementById('message-window');
const form = document.getElementById('input-form');
const input = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const clearButton = document.getElementById('clear-button');
const helpButton = document.getElementById('help-button');
const chatPopup = document.getElementById('chat-popup');
const feedbackModal = document.getElementById('feedback-modal');
const feedbackBackdrop = document.getElementById('feedback-backdrop');
const feedbackForm = document.getElementById('feedback-form');
const closeFeedbackBtn = document.getElementById('close-feedback-btn');
const feedbackTextarea = document.getElementById('feedback-textarea');
const languageSwitcher = document.getElementById('language-switcher');

// --- State ---
let chatHistory = [];
let currentFeedbackContext = {};
let translations = {};
let currentLang = 'zh';

console.log("index.js script loaded.");

// --- i18n Functions ---
async function loadLanguage(lang) {
    console.log(`Attempting to load language: ${lang}`);
    try {
        const response = await fetch(`/locales/${lang}.json`);
        if (!response.ok) {
            console.error(`Could not load ${lang}.json. Status: ${response.status}`);
            return;
        }
        translations = await response.json();
        currentLang = lang;
        applyTranslations();
        handleClearChat();
        console.log(`Language ${lang} loaded and applied successfully.`);
    } catch (error) {
        console.error('Failed to load language:', error);
    }
}

function applyTranslations() {
    console.log("Applying translations...");
    document.querySelectorAll('[data-i18n]').forEach(elem => {
        const key = elem.getAttribute('data-i18n');
        elem.innerHTML = translations[key] || elem.innerHTML;
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(elem => {
        const key = elem.getAttribute('data-i18n-placeholder');
        elem.placeholder = translations[key] || elem.placeholder;
    });
    document.querySelectorAll('[data-i18n-title]').forEach(elem => {
        const key = elem.getAttribute('data-i18n-title');
        elem.title = translations[key] || elem.title;
    });
    const pageTitleKey = 'title'; // Assuming 'title' key exists in your translation files
    const pageTitleElement = document.querySelector('.chat-title');
    if (pageTitleElement) {
        pageTitleElement.textContent = translations[pageTitleKey] || 'TCU Scholarship Q&A';
    }
    document.title = translations[pageTitleKey] || 'TCU Scholarship Q&A';
    console.log("Translations applied.");
}

// --- Event Listeners ---
form.addEventListener('submit', handleUserSubmit);
clearButton.addEventListener('click', handleClearChat);
helpButton.addEventListener('click', handleGetHelp);
input.addEventListener('input', () => {
    sendButton.disabled = input.value.trim() === '';
});
feedbackForm.addEventListener('submit', handleFeedbackSubmit);
closeFeedbackBtn.addEventListener('click', closeFeedbackModal);
feedbackBackdrop.addEventListener('click', closeFeedbackModal);
languageSwitcher.addEventListener('change', (e) => {
    console.log(`Language switcher changed to: ${e.target.value}`);
    loadLanguage(e.target.value);
});


// --- Initial State ---
sendButton.disabled = true;
loadLanguage(languageSwitcher.value); // Load initial language

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
    const thinkingText = translations['thinking_message'] || 'Thinking...';
    addMessage(`<span class="thinking">${thinkingText}</span>`, 'bot', thinkingMessageId);

    let fullAnswer = '';
    let isFirstChunk = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                history: chatHistory.slice(0, -1),
                lang: currentLang 
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
        const errorMsg = translations['error_message'] || 'Sorry, an error occurred while connecting. Please try again later.';
        const messageContainer = document.getElementById(thinkingMessageId);
        const messageEl = messageContainer?.querySelector('.bot-message');
        if (messageEl) {
            messageEl.innerHTML = errorMsg;
        }
        chatHistory.push({ role: 'assistant', content: errorMsg });
    } finally {
        input.focus();
    }
}

function handleClearChat() {
    msgWindow.innerHTML = '';
    const initialMessage = translations['initial_bot_message'] || 'Hello! How can I help you?';
    addMessage(initialMessage, 'bot');
    chatHistory = [{'role': 'assistant', 'content': initialMessage}];
    
    if (!msgWindow.querySelector('.example-questions-container')) {
        const exampleContainer = document.createElement('div');
        exampleContainer.className = 'example-questions-container';
        exampleContainer.innerHTML = `
            <div class="example-question" onclick="askQuestion('${translations['example_question_1']}')">${translations['example_question_1']}</div>
            <div class="example-question" onclick="askQuestion('${translations['example_question_2']}')">${translations['example_question_2']}</div>
            <div class="example-question" onclick="askQuestion('${translations['example_question_3']}')">${translations['example_question_3']}</div>
            <div class="example-question" onclick="askQuestion('${translations['example_question_4']}')">${translations['example_question_4']}</div>
            <div class="example-question" onclick="askQuestion('${translations['example_question_5']}')">${translations['example_question_5']}</div>
        `;
        msgWindow.appendChild(exampleContainer);
    }
}

function handleGetHelp() {
    alert(translations['help_alert'] || 'Contact info not available.');
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
        let contextsHTML = `<h4>${translations['reference_title'] || 'References:'}</h4>`;
        contexts.forEach(ctx => {
            const fileName = ctx.source_file || translations['unknown_source'] || 'Unknown source';
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
            <button class="feedback-btn like-btn" title="${translations['like_button_title'] || 'Satisfied'}"><i class="fa-regular fa-thumbs-up fa-lg" style="color: #adb1b9;"></i></button>
            <button class="feedback-btn dislike-btn" title="${translations['dislike_button_title'] || 'Dissatisfied'}"><i class="fa-regular fa-thumbs-down fa-lg" style="color: #adb1b9;"></i></button>
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

    if (isSelected) {
        target.classList.remove('selected');
        await sendFeedback(logId, null, null); 
    } else {
        if (isDislike) {
            likeBtn.classList.remove('selected');
        } else {
            dislikeBtn.classList.remove('selected');
        }
        
        target.classList.add('selected');

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
        // alert(translations['feedback_error_alert'] || 'Sorry, there was an error submitting your feedback.');
    }
}