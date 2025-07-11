from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON, Enum, Date, DECIMAL, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import os
import feedparser
import requests
import openai
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import json
from typing import Literal

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:root@localhost/podcast_analytics")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your-claude-key")

# Validate required environment variables
if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-key":
    logging.warning("OpenAI API key not properly configured")

if not SECRET_KEY or SECRET_KEY == "your-secret-key-here":
    logging.warning("Insecure secret key configuration")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# FastAPI app
app = FastAPI(
    title="Podcast Analytics Platform",
    version="1.0.0",
    description="A comprehensive platform for podcast analytics and insights",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-production-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))
    name = Column(String(100))
    role = Column(Enum('podcaster', 'advertiser', 'admin'), default='podcaster')
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    podcasts = relationship("Podcast", back_populates="owner")
    subscriptions = relationship("Subscription", back_populates="user")

class Podcast(Base):
    __tablename__ = "podcasts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    description = Column(Text)
    rss_feed_url = Column(String(500))
    image_url = Column(String(500))
    author = Column(String(255))
    category = Column(String(100))
    language = Column(String(10), default='en')
    created_by = Column(Integer, ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="podcasts")
    episodes = relationship("Episode", back_populates="podcast")
    subscriptions = relationship("Subscription", back_populates="podcast")

class Episode(Base):
    __tablename__ = "episodes"
    
    id = Column(Integer, primary_key=True, index=True)
    podcast_id = Column(Integer, ForeignKey("podcasts.id"))
    title = Column(String(255))
    description = Column(Text)
    audio_url = Column(String(500))
    duration = Column(Integer)
    file_size = Column(BigInteger)
    episode_number = Column(Integer, nullable=True)
    season_number = Column(Integer, nullable=True)
    pub_date = Column(DateTime)
    guid = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    podcast = relationship("Podcast", back_populates="episodes")
    analytics = relationship("Analytics", back_populates="episode")
    ai_analysis = relationship("AIAnalysis", back_populates="episode")

class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"))
    date = Column(Date)
    downloads = Column(Integer, default=0)
    listens = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    completion_rate = Column(DECIMAL(5,2))
    avg_listen_duration = Column(Integer)
    unique_listeners = Column(Integer, default=0)
    
    episode = relationship("Episode", back_populates="analytics")

class AIAnalysis(Base):
    __tablename__ = "ai_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"))
    summary = Column(Text)
    keywords = Column(JSON)
    topics = Column(JSON)
    sentiment = Column(Enum('positive', 'neutral', 'negative'))
    sentiment_score = Column(DECIMAL(3,2))
    transcript = Column(Text)
    virality_score = Column(DECIMAL(3,2))
    predicted_performance = Column(Enum('low', 'medium', 'high'))
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    episode = relationship("Episode", back_populates="ai_analysis")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    podcast_id = Column(Integer, ForeignKey("podcasts.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="subscriptions")
    podcast = relationship("Podcast", back_populates="subscriptions")

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "podcaster"

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v

    @validator('role')
    def validate_role(cls, v):
        if v not in ['podcaster', 'advertiser', 'admin']:
            raise ValueError("Invalid role")
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    name: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class PodcastCreate(BaseModel):
    title: str
    description: str
    rss_feed_url: str
    category: str

class PodcastResponse(BaseModel):
    id: int
    title: str
    description: str
    rss_feed_url: str
    image_url: Optional[str]
    author: str
    category: str
    language: str
    is_active: bool
    created_at: datetime
    last_updated: Optional[datetime]
    
    class Config:
        from_attributes = True

class EpisodeResponse(BaseModel):
    id: int
    podcast_id: int
    title: str
    description: str
    audio_url: str
    duration: int
    file_size: int
    episode_number: Optional[int]
    season_number: Optional[int]
    pub_date: datetime
    guid: str
    
    class Config:
        from_attributes = True

class AnalyticsResponse(BaseModel):
    id: int
    episode_id: int
    date: str
    downloads: int
    listens: int
    shares: int
    completion_rate: Optional[float]
    avg_listen_duration: Optional[int]
    unique_listeners: int
    
    class Config:
        from_attributes = True

class AIAnalysisResponse(BaseModel):
    id: int
    episode_id: int
    summary: str
    keywords: List[str]
    topics: List[str]
    sentiment: Literal['positive', 'neutral', 'negative']
    sentiment_score: float
    virality_score: float
    predicted_performance: Literal['low', 'medium', 'high']
    processed_at: datetime
    
    class Config:
        from_attributes = True

class DashboardOverviewResponse(BaseModel):
    total_podcasts: int
    total_episodes: int
    total_downloads: int
    avg_completion_rate: float
    top_podcasts: List[Dict[str, Any]]
    recent_episodes: List[Dict[str, Any]]

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(email: str = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update last login time
    user.last_login = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return user

# RSS Feed Parser
def parse_rss_feed(feed_url: str) -> Dict[str, Any]:
    """Parse RSS feed and extract podcast/episode information"""
    try:
        feed = feedparser.parse(feed_url)
        
        if feed.bozo:  # indicates parsing error
            raise ValueError(f"Feed parsing error: {feed.bozo_exception}")
            
        podcast_info = {
            'title': feed.feed.get('title', 'Untitled Podcast'),
            'description': feed.feed.get('description', 'No description available'),
            'author': feed.feed.get('author', 'Unknown author'),
            'image_url': feed.feed.get('image', {}).get('href', ''),
            'language': feed.feed.get('language', 'en')
        }
        
        episodes = []
        for entry in feed.entries:
            episode = {
                'title': entry.get('title', ''),
                'description': entry.get('summary', ''),
                'guid': entry.get('guid', ''),
                'pub_date': entry.get('published_parsed'),
                'audio_url': '',
                'duration': 0,
                'file_size': 0
            }
            
            # Extract audio URL
            for link in entry.get('links', []):
                if link.get('type', '').startswith('audio/'):
                    episode['audio_url'] = link.get('href', '')
                    episode['file_size'] = int(link.get('length', 0))
                    break
            
            # Extract duration from iTunes tags
            if hasattr(entry, 'itunes_duration'):
                episode['duration'] = parse_duration(entry.itunes_duration)
            
            episodes.append(episode)
        
        return {'podcast': podcast_info, 'episodes': episodes}
    except Exception as e:
        logging.error(f"Error parsing RSS feed {feed_url}: {str(e)}")
        return {'podcast': {}, 'episodes': []}

def parse_duration(duration_str: str) -> int:
    """Convert duration string to seconds"""
    try:
        parts = duration_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return int(parts[0])
    except:
        return 0

# AI Analysis Functions
def analyze_episode_with_ai(episode_text: str) -> Dict[str, Any]:
    """Analyze episode content using OpenAI"""
    try:
        if not episode_text.strip():
            raise ValueError("Empty episode text provided")
            
        openai.api_key = OPENAI_API_KEY
        
        prompt = f"""
        Analyze the following podcast episode content and provide:
        1. A 3-sentence summary
        2. 5-10 relevant keywords
        3. 3-5 main topics
        4. Sentiment analysis (positive/neutral/negative) with score (-1 to 1)
        5. Virality score (0 to 1) based on engagement potential
        6. Predicted performance (low/medium/high)
        
        Episode content:
        {episode_text[:4000]}  # Limit to avoid token limits
        
        Return the analysis in JSON format with keys: summary, keywords, topics, sentiment, sentiment_score, virality_score, predicted_performance
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Properly parse the JSON response
        try:
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            # Validate required fields
            required_fields = ['summary', 'keywords', 'topics', 'sentiment', 
                             'sentiment_score', 'virality_score', 'predicted_performance']
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing field in AI response: {field}")
                    
            return analysis
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error parsing AI response: {str(e)}")
            raise
    except Exception as e:
        logging.error(f"Error in AI analysis: {str(e)}")
        return {
            'summary': "Analysis unavailable due to error",
            'keywords': [],
            'topics': [],
            'sentiment': "neutral",
            'sentiment_score': 0.0,
            'virality_score': 0.0,
            'predicted_performance': "low"
        }

# API Routes
@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    password_hash = pwd_context.hash(user.password)
    
    # Create user
    db_user = User(
        email=user.email,
        password_hash=password_hash,
        name=user.name,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create token
    access_token = create_access_token(data={"sub": user.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.post("/auth/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    
    if not db_user or not pwd_context.verify(user.password, db_user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/podcasts", response_model=List[PodcastResponse])
async def get_podcasts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    podcasts = db.query(Podcast)\
        .filter(Podcast.created_by == current_user.id)\
        .offset(skip)\
        .limit(limit)\
        .all()
    return podcasts

@app.post("/podcasts", response_model=PodcastResponse)
async def create_podcast(
    podcast: PodcastCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Parse RSS feed
        feed_data = parse_rss_feed(podcast.rss_feed_url)
        
        # Create podcast
        db_podcast = Podcast(
            title=podcast.title,
            description=podcast.description,
            rss_feed_url=podcast.rss_feed_url,
            category=podcast.category,
            created_by=current_user.id,
            author=feed_data['podcast'].get('author', ''),
            image_url=feed_data['podcast'].get('image_url', ''),
            language=feed_data['podcast'].get('language', 'en')
        )
        db.add(db_podcast)
        db.commit()
        db.refresh(db_podcast)
        
        # Create episodes
        for episode_data in feed_data['episodes']:
            try:
                pub_date = datetime(*episode_data['pub_date'][:6]) if episode_data['pub_date'] else datetime.utcnow()
                
                db_episode = Episode(
                    podcast_id=db_podcast.id,
                    title=episode_data['title'],
                    description=episode_data['description'],
                    audio_url=episode_data['audio_url'],
                    duration=episode_data['duration'],
                    file_size=episode_data['file_size'],
                    guid=episode_data['guid'],
                    pub_date=pub_date
                )
                db.add(db_episode)
            except Exception as e:
                db.rollback()
                logging.error(f"Error creating episode: {str(e)}")
                continue
        
        db.commit()
        return db_podcast
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/podcasts/{podcast_id}/episodes", response_model=List[EpisodeResponse])
async def get_episodes(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    podcast = db.query(Podcast)\
        .filter(Podcast.id == podcast_id, Podcast.created_by == current_user.id)\
        .first()
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")
    
    episodes = db.query(Episode)\
        .filter(Episode.podcast_id == podcast_id)\
        .offset(skip)\
        .limit(limit)\
        .all()
    return episodes

@app.get("/episodes/{episode_id}/analytics", response_model=AnalyticsResponse)
async def get_episode_analytics(
    episode_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    episode = db.query(Episode)\
        .join(Podcast)\
        .filter(
            Episode.id == episode_id,
            Podcast.created_by == current_user.id
        )\
        .first()
    
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    analytics = db.query(Analytics)\
        .filter(Analytics.episode_id == episode_id)\
        .first()
    
    if not analytics:
        # Create dummy analytics for demo
        analytics = Analytics(
            episode_id=episode_id,
            date=datetime.utcnow().date(),
            downloads=100,
            listens=85,
            shares=12,
            completion_rate=78.5,
            avg_listen_duration=episode.duration * 0.8,
            unique_listeners=82
        )
        db.add(analytics)
        db.commit()
    
    return analytics

@app.get("/episodes/{episode_id}/ai-analysis", response_model=AIAnalysisResponse)
async def get_episode_ai_analysis(
    episode_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    episode = db.query(Episode)\
        .join(Podcast)\
        .filter(
            Episode.id == episode_id,
            Podcast.created_by == current_user.id
        )\
        .first()
    
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    ai_analysis = db.query(AIAnalysis)\
        .filter(AIAnalysis.episode_id == episode_id)\
        .first()
    
    if not ai_analysis:
        # Generate AI analysis
        analysis_data = analyze_episode_with_ai(episode.description)
        
        ai_analysis = AIAnalysis(
            episode_id=episode_id,
            summary=analysis_data['summary'],
            keywords=analysis_data['keywords'],
            topics=analysis_data['topics'],
            sentiment=analysis_data['sentiment'],
            sentiment_score=analysis_data['sentiment_score'],
            virality_score=analysis_data['virality_score'],
            predicted_performance=analysis_data['predicted_performance']
        )
        db.add(ai_analysis)
        db.commit()
    
    return ai_analysis

@app.get("/dashboard/overview", response_model=DashboardOverviewResponse)
async def get_dashboard_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard overview statistics"""
    total_podcasts = db.query(Podcast)\
        .filter(Podcast.created_by == current_user.id)\
        .count()
    
    total_episodes = db.query(Episode)\
        .join(Podcast)\
        .filter(Podcast.created_by == current_user.id)\
        .count()
    
    # Get total downloads (mock data for demo)
    total_downloads = db.query(Analytics)\
        .join(Episode)\
        .join(Podcast)\
        .filter(Podcast.created_by == current_user.id)\
        .count() * 150  # Mock calculation
    
    # Get top podcasts (mock data)
    top_podcasts = db.query(Podcast)\
        .filter(Podcast.created_by == current_user.id)\
        .order_by(Podcast.created_at.desc())\
        .limit(3)\
        .all()
    
    # Get recent episodes (mock data)
    recent_episodes = db.query(Episode)\
        .join(Podcast)\
        .filter(Podcast.created_by == current_user.id)\
        .order_by(Episode.pub_date.desc())\
        .limit(5)\
        .all()
    
    return {
        "total_podcasts": total_podcasts,
        "total_episodes": total_episodes,
        "total_downloads": total_downloads,
        "avg_completion_rate": 76.5,
        "top_podcasts": top_podcasts,
        "recent_episodes": recent_episodes
    }

# Background task for RSS feed updates
def update_rss_feeds():
    """Background task to update RSS feeds"""
    db = SessionLocal()
    try:
        podcasts = db.query(Podcast).filter(Podcast.is_active == True).all()
        for podcast in podcasts:
            try:
                feed_data = parse_rss_feed(podcast.rss_feed_url)
                
                # Update existing episodes and add new ones
                for episode_data in feed_data['episodes']:
                    existing_episode = db.query(Episode)\
                        .filter(Episode.guid == episode_data['guid'])\
                        .first()
                    
                    if not existing_episode:
                        pub_date = datetime(*episode_data['pub_date'][:6]) if episode_data['pub_date'] else datetime.utcnow()
                        
                        new_episode = Episode(
                            podcast_id=podcast.id,
                            title=episode_data['title'],
                            description=episode_data['description'],
                            audio_url=episode_data['audio_url'],
                            duration=episode_data['duration'],
                            file_size=episode_data['file_size'],
                            guid=episode_data['guid'],
                            pub_date=pub_date
                        )
                        db.add(new_episode)
                
                db.commit()
            except Exception as e:
                db.rollback()
                logging.error(f"Error updating podcast {podcast.id}: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"Error in RSS feed update task: {str(e)}")
    finally:
        db.close()

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_rss_feeds, trigger="interval", hours=1)
scheduler.start()

# Shutdown handler
@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

# Create tables
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
