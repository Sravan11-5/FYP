# API Keys Setup Guide

## 🎬 TMDB API Key

### Step 1: Create TMDB Account
1. Go to: https://www.themoviedb.org/signup
2. Sign up for a free account
3. Verify your email address

### Step 2: Request API Key
1. Go to: https://www.themoviedb.org/settings/api
2. Click "Request an API Key"
3. Choose "Developer" option
4. Fill in the application form:
   - **Application Name**: Telugu Movie Recommendation System
   - **Application URL**: http://localhost:8000 (or your domain)
   - **Application Summary**: AI-powered recommendation system for Telugu movies

### Step 3: Get Your API Key
1. Once approved (instant for most cases), you'll see your API Key (v3 auth)
2. Copy the API Key
3. Add to `.env` file:
   ```
   TMDB_API_KEY="your_api_key_here"
   ```

---

## 🐦 Twitter API Keys

### Step 1: Create Twitter Developer Account
1. Go to: https://developer.twitter.com/
2. Sign up or log in with your Twitter account
3. Apply for "Elevated" access (required for Twitter API v2)

### Step 2: Create a Project and App
1. Go to: https://developer.twitter.com/en/portal/projects-and-apps
2. Click "Create Project"
3. Fill in details:
   - **Project Name**: Telugu Movie Reviews
   - **Use Case**: Academic research or Student project
   - **Project Description**: Collecting Telugu language movie reviews for sentiment analysis

4. Click "Create App" within your project
   - **App Name**: telugu-movie-reviews (must be unique)

### Step 3: Get Your API Keys
1. After creating the app, you'll see:
   - **API Key** (Consumer Key)
   - **API Key Secret** (Consumer Secret)
   - **Bearer Token**

2. **IMPORTANT**: Save these immediately! You won't see them again.

3. If you need to regenerate:
   - Go to your app settings
   - Navigate to "Keys and tokens" tab
   - Click "Regenerate" for any key

### Step 4: Enable Read Access
1. In your app settings, go to "Settings" tab
2. Under "App permissions", ensure it's set to "Read" (default)
3. This is sufficient for searching tweets

### Step 5: Add to .env File
```env
# Twitter API Keys
TWITTER_API_KEY="your_api_key_here"
TWITTER_API_SECRET="your_api_secret_here"
TWITTER_BEARER_TOKEN="your_bearer_token_here"
```

---

## Complete .env File Example

```env
# API Keys
GOOGLE_API_KEY="AIzaSyDMcu-f4gwv0z1vicxR9uvM1Rqm1DI3-cU"

# MongoDB Configuration (MongoDB Atlas Cloud)
MONGODB_URL="mongodb+srv://ProjectRefresh:projectrefresh@cluster0.oj8eytc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB_NAME="telugu_movie_recommender"

# Twitter API Keys (for review collection)
TWITTER_API_KEY="your_api_key_from_step_3"
TWITTER_API_SECRET="your_api_secret_from_step_3"
TWITTER_BEARER_TOKEN="your_bearer_token_from_step_3"

# TMDB API Key (for movie data)
TMDB_API_KEY="your_tmdb_api_key_from_step_3"
```

---

## Testing Your API Keys

Once you've added the keys to `.env`, run the test script:

```bash
python test_api_clients.py
```

This will verify:
- ✅ TMDB API connection
- ✅ Movie search functionality
- ✅ Twitter API connection
- ✅ Telugu tweet search

---

## Troubleshooting

### TMDB API Issues
- **401 Unauthorized**: Check if API key is correct
- **404 Not Found**: Verify the endpoint URL
- **Rate Limit**: Free tier allows 1000 requests/day

### Twitter API Issues
- **403 Forbidden**: You may need "Elevated" access
  - Go to developer portal
  - Request elevated access (takes 1-2 days)
- **429 Too Many Requests**: Rate limit exceeded
  - Free tier: 500k tweets/month
  - Wait 15 minutes and try again
- **401 Unauthorized**: Check Bearer Token is correct

### Common Mistakes
1. **Extra spaces** in API keys (copy-paste issue)
2. **Quotes** - Use double quotes in .env: `KEY="value"`
3. **Missing credentials** - All three Twitter keys are needed
4. **Wrong app permissions** - Ensure "Read" access is enabled

---

## API Rate Limits

### TMDB API
- **Free Tier**: 1,000 requests per day
- **Request Limit**: 40 requests every 10 seconds

### Twitter API
- **Essential (Free)**:
  - 500,000 tweets per month
  - 1,500 tweets per 15 minutes per endpoint
- **Elevated (Free)**:
  - 2,000,000 tweets per month
  - Higher rate limits

---

## Security Best Practices

1. ✅ **Never commit .env file** to Git
   - Already in `.gitignore`

2. ✅ **Use environment variables** 
   - Never hardcode keys in code

3. ✅ **Regenerate keys** if accidentally exposed
   - TMDB: https://www.themoviedb.org/settings/api
   - Twitter: Developer portal → App settings → Keys and tokens

4. ✅ **Use different keys** for development and production

---

## Need Help?

- **TMDB API Docs**: https://developers.themoviedb.org/3
- **Twitter API Docs**: https://developer.twitter.com/en/docs/twitter-api
- **Support**: Contact respective API support teams

---

**Status**: Task 4 Complete - Ready for testing once API keys are configured! 🎉
