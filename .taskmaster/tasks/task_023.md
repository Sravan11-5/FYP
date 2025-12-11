# Task ID: 23

**Title:** Display recommendations with movie details

**Status:** pending

**Dependencies:** 22

**Priority:** medium

**Description:** Display the recommended movies with movie details and explanations on the results page.

**Details:**

1. Show movie poster, title, genre, rating, sentiment score.
2. Provide explanation for each recommendation.
3. Click to view detailed movie information.

**Test Strategy:**

1. Verify the movie details and explanations are displayed correctly.
2. Check the click to view detailed movie information functionality.

## Subtasks

### 23.1. Implement Movie Details Display

**Status:** pending  
**Dependencies:** 23.23  

Implement the display of movie details including poster, title, genre, rating, and sentiment score on the results page.

**Details:**

Fetch movie details from the database and display them in a structured format using HTML/CSS. Ensure responsiveness for different screen sizes. Use appropriate UI elements for each detail.

### 23.2. Provide Recommendation Explanations

**Status:** pending  
**Dependencies:** 23.23, 23.18  

Generate and display explanations for each movie recommendation, such as sentiment analysis results or genre similarities.

**Details:**

Integrate with the recommendation explanation service to fetch explanations. Display the explanations clearly next to each movie recommendation. Ensure explanations are informative and easy to understand.

### 23.3. Implement Click to View Detailed Information

**Status:** pending  
**Dependencies:** 23.23  

Implement the functionality to allow users to click on a movie recommendation to view detailed movie information.

**Details:**

Add a click event listener to each movie recommendation. Upon clicking, redirect the user to a detailed movie information page. Pass the movie ID to the detailed page. Ensure the detailed page displays all relevant information.
