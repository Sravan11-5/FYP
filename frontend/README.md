# Telugu Movie Recommendation System - Frontend

## Overview
Modern, responsive web interface for the Telugu Movie Recommendation System built with HTML5, CSS3, and vanilla JavaScript.

## Features

### ✅ Completed (Task 3)
- **Telugu & English bilingual interface**
- **Responsive design** - Works on mobile, tablet, and desktop
- **Modern UI/UX** - Clean gradient design with smooth animations
- **Search functionality** - With loading indicators and error handling
- **Accessibility** - ARIA labels, keyboard shortcuts, semantic HTML
- **Performance** - Optimized CSS, lazy loading images

## File Structure

```
frontend/
├── index.html          # Main HTML file with search interface
├── css/
│   └── styles.css      # Complete responsive styles
└── js/
    └── app.js          # Search logic and API integration
```

## Key Features

### 1. Search Interface
- Telugu and English input support
- Real-time validation
- Loading indicators
- Error handling

### 2. Responsive Design
- Mobile-first approach
- Breakpoints: 576px, 768px, 992px, 1200px
- Flexible grid layout
- Touch-friendly buttons

### 3. Typography
- **Telugu**: Noto Sans Telugu font (Google Fonts)
- **English**: Poppins font (Google Fonts)
- Optimized for readability

### 4. Color Scheme
- Primary: Blue gradient (#1e3a8a → #1e40af)
- Accent: Orange (#f59e0b)
- Background: Purple gradient (#667eea → #764ba2)

### 5. Keyboard Shortcuts
- `/` - Focus search input
- `Escape` - Clear search and results

## Usage

### Development
Simply open `index.html` in a browser:
```bash
# Windows
start index.html

# Or serve with Python
python -m http.server 8080
# Then open http://localhost:8080
```

### Production
Serve the `frontend` folder with any static web server:
- Nginx
- Apache
- Netlify
- Vercel
- GitHub Pages

## API Integration (Coming Soon)

The frontend is ready to integrate with the FastAPI backend:

```javascript
// Current configuration in app.js
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000'
};
```

### Expected API Endpoints
- `POST /api/search` - Search for movies
- `GET /api/movie/{id}` - Get movie details
- `POST /api/recommend` - Get recommendations

## Browser Support
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Performance Optimizations
- CSS variables for theming
- Smooth transitions (0.3s)
- Lazy loading for images
- Debounced search input
- Minimal DOM manipulation

## Accessibility
- ✅ ARIA labels
- ✅ Keyboard navigation
- ✅ Screen reader support
- ✅ Focus indicators
- ✅ Semantic HTML5

## Next Steps (Tasks 4-5)
1. **Backend Integration**
   - Connect to FastAPI endpoints
   - Handle API responses
   - Display real movie data

2. **TMDB API Integration**
   - Fetch movie posters
   - Get movie details
   - Display ratings

3. **Twitter Integration**
   - Show Telugu reviews
   - Display sentiment scores

4. **Recommendations**
   - ML-based suggestions
   - Similar movies display

## Testing Checklist

### ✅ Visual Testing
- [x] Responsive on mobile (320px - 768px)
- [x] Responsive on tablet (768px - 1024px)
- [x] Responsive on desktop (1024px+)
- [x] Telugu text displays correctly
- [x] Gradient backgrounds render properly
- [x] Icons display correctly

### ✅ Functional Testing
- [x] Search form validation
- [x] Loading indicator shows/hides
- [x] Error messages display
- [x] Keyboard shortcuts work
- [x] Buttons are clickable

### ⏳ Integration Testing (Pending Backend)
- [ ] API calls succeed
- [ ] Results display correctly
- [ ] Recommendations load
- [ ] Error handling for API failures

## Technologies Used
- **HTML5** - Semantic markup
- **CSS3** - Flexbox, Grid, Custom Properties
- **JavaScript (ES6+)** - Async/await, Modules
- **Google Fonts** - Noto Sans Telugu, Poppins

## Credits
Built for the Telugu Movie Recommendation System using Agentic AI and Deep Learning.

---

**Status**: ✅ Task 3 Complete - Ready for backend integration
