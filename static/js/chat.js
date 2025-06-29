document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.querySelector('.chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const typingIndicator = document.querySelector('.typing-indicator');
    
    // Focus input on page load
    userInput.focus();
    
    // Add welcome message
    addBotMessage("Hello! I'm your Jupiter FAQ assistant. How can I help you today?");
    
    // Handle send button click
    sendButton.addEventListener('click', sendMessage);
    
    // Handle Enter key press
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addUserMessage(message);
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to server
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        })
        .then(response => response.json())
        .then(data => {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add bot response
            addBotResponse(data);
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            addBotMessage("Sorry, I encountered an error. Please try again.");
        });
    }
    
    function addUserMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.textContent = message;
        chatContainer.appendChild(messageDiv);
        scrollToBottom();
    }
    
    function addBotMessage(message, source = '', relatedQueries = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        let messageHTML = `<div>${message}</div>`;
        
        if (source) {
            messageHTML += `<div class="source">Source: <a href="${source}" target="_blank">${source}</a></div>`;
        }
        
        if (relatedQueries && relatedQueries.length > 0) {
            messageHTML += '<div class="related-queries"><h4>Related Queries:</h4>';
            relatedQueries.forEach(query => {
                messageHTML += `<span class="related-query" onclick="this.parentNode.parentNode.querySelector('input').value='${query.replace(/'/g, "\\'")}'; this.parentNode.parentNode.querySelector('button').click()">${query}</span> `;
            });
            messageHTML += '</div>';
        }
        
        messageDiv.innerHTML = messageHTML;
        chatContainer.appendChild(messageDiv);
        scrollToBottom();
    }
    
    function addBotResponse(data) {
        const { answer, source, related_queries, response_time } = data;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        let messageHTML = `<div>${answer}</div>`;
        
        // Add response time
        if (response_time) {
            const seconds = (response_time / 1000).toFixed(2);
            messageHTML += `<div class="response-time">Response time: ${seconds}s</div>`;
        }
        
        if (source) {
            messageHTML += `<div class="source">Source: <a href="${source}" target="_blank">${source}</a></div>`;
        }
        
        if (related_queries && related_queries.length > 0) {
            messageHTML += '<div class="related-queries"><h4>Related Queries:</h4>';
            related_queries.forEach(query => {
                messageHTML += `<span class="related-query" onclick="document.getElementById('user-input').value='${query.replace(/'/g, "\\'")}'; document.getElementById('send-button').click()">${query}</span> `;
            });
            messageHTML += '</div>';
        }
        
        messageDiv.innerHTML = messageHTML;
        chatContainer.appendChild(messageDiv);
        scrollToBottom();
    }
    
    function showTypingIndicator() {
        typingIndicator.style.display = 'block';
        scrollToBottom();
    }
    
    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }
    
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Make functions available globally for related queries
    window.addUserMessage = addUserMessage;
    window.addBotMessage = addBotMessage;
});
