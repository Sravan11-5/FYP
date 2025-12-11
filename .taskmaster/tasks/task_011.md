# Task ID: 11

**Title:** Collect and prepare Telugu movie reviews dataset

**Status:** pending

**Dependencies:** None

**Priority:** medium

**Description:** Gather and preprocess a dataset of Telugu movie reviews for training the Siamese Network.

**Details:**

1. Scrape Telugu movie reviews from various sources (if necessary, supplement with existing datasets).
2. Clean the data by removing irrelevant characters, HTML tags, and noise.
3. Tokenize the reviews using a Telugu tokenizer.
4. Split the dataset into training, validation, and testing sets (80/10/10 split).

**Test Strategy:**

1. Verify the dataset contains a sufficient number of Telugu movie reviews (50,000+).
2. Check the quality of the cleaned and tokenized data.
3. Validate the dataset split.

## Subtasks

### 11.1. Scrape Telugu movie reviews

**Status:** pending  
**Dependencies:** None  

Gather Telugu movie reviews from various online sources, supplementing with existing datasets if needed.

**Details:**

Implement web scraping scripts to extract reviews from sites like Idlebrain, Greatandhra, and others. Handle pagination and data extraction efficiently.

### 11.2. Clean Telugu movie review data

**Status:** pending  
**Dependencies:** 11.1  

Clean the scraped data by removing irrelevant characters, HTML tags, and noise from the reviews.

**Details:**

Use regular expressions and string manipulation techniques to remove HTML tags, special characters, and irrelevant information. Handle Telugu-specific character encoding issues.

### 11.3. Tokenize Telugu movie reviews

**Status:** pending  
**Dependencies:** 11.2  

Tokenize the cleaned Telugu movie reviews using a suitable Telugu tokenizer.

**Details:**

Implement or integrate a Telugu tokenizer (e.g., using Indic NLP Library) to split the reviews into individual tokens. Handle compound words and morphological variations.

### 11.4. Split dataset into training, validation, and testing sets

**Status:** pending  
**Dependencies:** 11.3  

Divide the tokenized dataset into training, validation, and testing sets with an 80/10/10 split.

**Details:**

Implement a function to split the dataset randomly into training (80%), validation (10%), and testing (10%) sets. Ensure data is shuffled before splitting.
