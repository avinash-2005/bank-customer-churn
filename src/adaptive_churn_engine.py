"""
Adaptive Churn Scoring Engine
Dynamically adjusts churn predictions based on real-time behavior patterns, temporal trends, and customer segments
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class AdaptiveChurnEngine:
    """Advanced adaptive churn scoring system with dynamic risk assessment"""
    
    def __init__(self):
        self.behavior_history = {}
        self.temporal_patterns = {}
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7,
            'high': 0.9
        }
        self.segment_models = {}
        self.adaptive_weights = {
            'recent_behavior': 0.4,
            'historical_pattern': 0.3,
            'sentiment_trend': 0.2,
            'segment_baseline': 0.1
        }
        self.scaler = StandardScaler()
        
    def track_customer_behavior(self, customer_id, behavior_data):
        """Track customer behavior over time for pattern analysis"""
        timestamp = datetime.now()
        
        if customer_id not in self.behavior_history:
            self.behavior_history[customer_id] = []
        
        # Add new behavior data point
        behavior_entry = {
            'timestamp': timestamp,
            'login_frequency': behavior_data.get('login_frequency', 0),
            'session_duration': behavior_data.get('session_duration', 0),
            'inactivity_days': behavior_data.get('inactivity_days', 0),
            'transaction_count': behavior_data.get('transaction_count', 0),
            'account_balance': behavior_data.get('account_balance', 0),
            'credit_score': behavior_data.get('credit_score', 0),
            'sentiment_score': behavior_data.get('sentiment_score', 0.5)
        }
        
        self.behavior_history[customer_id].append(behavior_entry)
        
        # Keep only last 30 days of data
        cutoff_date = timestamp - timedelta(days=30)
        self.behavior_history[customer_id] = [
            entry for entry in self.behavior_history[customer_id]
            if entry['timestamp'] > cutoff_date
        ]
        
        return behavior_entry
    
    def analyze_temporal_patterns(self, customer_id):
        """Analyze temporal behavior patterns for adaptive scoring"""
        if customer_id not in self.behavior_history or len(self.behavior_history[customer_id]) < 3:
            return {'trend': 'stable', 'volatility': 0, 'consistency': 0.5}
        
        history = self.behavior_history[customer_id]
        
        # Calculate trends
        recent_data = history[-7:] if len(history) >= 7 else history
        older_data = history[-14:-7] if len(history) >= 14 else history[:-len(recent_data)]
        
        if len(recent_data) > 0 and len(older_data) > 0:
            recent_avg = np.mean([d['login_frequency'] for d in recent_data])
            older_avg = np.mean([d['login_frequency'] for d in older_data])
            
            if recent_avg > older_avg * 1.2:
                trend = 'improving'
            elif recent_avg < older_avg * 0.8:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Calculate volatility (standard deviation of login frequency)
        login_freqs = [d['login_frequency'] for d in history]
        volatility = np.std(login_freqs) if len(login_freqs) > 1 else 0
        
        # Calculate consistency (inverse of volatility)
        consistency = 1 / (1 + volatility)
        
        return {
            'trend': trend,
            'volatility': volatility,
            'consistency': consistency,
            'recent_avg': recent_avg if len(recent_data) > 0 else 0
        }
    
    def calculate_adaptive_features(self, customer_id, current_data):
        """Create adaptive features based on behavior history"""
        features = {}
        
        # Get temporal patterns
        temporal = self.analyze_temporal_patterns(customer_id)
        
        # Behavioral change indicators
        if customer_id in self.behavior_history and len(self.behavior_history[customer_id]) > 1:
            last_entry = self.behavior_history[customer_id][-2]  # Previous entry
            current_entry = self.behavior_history[customer_id][-1]  # Current entry
            
            # Recent behavior change
            features['login_frequency_change'] = (
                current_entry['login_frequency'] - last_entry['login_frequency']
            )
            features['session_duration_change'] = (
                current_entry['session_duration'] - last_entry['session_duration']
            )
            features['inactivity_change'] = (
                current_entry['inactivity_days'] - last_entry['inactivity_days']
            )
            features['transaction_change'] = (
                current_entry['transaction_count'] - last_entry['transaction_count']
            )
        else:
            features['login_frequency_change'] = 0
            features['session_duration_change'] = 0
            features['inactivity_change'] = 0
            features['transaction_change'] = 0
        
        # Temporal pattern features
        features['temporal_trend_score'] = {
            'improving': 0.2,
            'stable': 0.0,
            'declining': -0.3
        }.get(temporal['trend'], 0.0)
        
        features['behavioral_volatility'] = temporal['volatility']
        features['behavioral_consistency'] = temporal['consistency']
        
        # Engagement momentum
        if customer_id in self.behavior_history and len(self.behavior_history[customer_id]) >= 3:
            recent_engagement = [
                d['login_frequency'] * d['session_duration'] 
                for d in self.behavior_history[customer_id][-3:]
            ]
            features['engagement_momentum'] = np.mean(recent_engagement)
        else:
            features['engagement_momentum'] = current_data.get('login_frequency', 0) * current_data.get('session_duration', 0)
        
        # Sentiment trend analysis
        if customer_id in self.behavior_history and len(self.behavior_history[customer_id]) >= 2:
            recent_sentiments = [d['sentiment_score'] for d in self.behavior_history[customer_id][-5:]]
            features['sentiment_trend'] = np.mean(recent_sentiments[-3:]) - np.mean(recent_sentiments[:-3]) if len(recent_sentiments) >= 4 else 0
        else:
            features['sentiment_trend'] = 0
        
        return features
    
    def segment_customers(self, customer_data):
        """Segment customers for personalized risk assessment"""
        segments = []
        
        # Value-based segmentation
        if customer_data.get('account_balance', 0) > 50000:
            segments.append('high_value')
        elif customer_data.get('account_balance', 0) > 10000:
            segments.append('medium_value')
        else:
            segments.append('low_value')
        
        # Activity-based segmentation
        if customer_data.get('login_frequency', 0) > 20:
            segments.append('high_activity')
        elif customer_data.get('login_frequency', 0) > 5:
            segments.append('medium_activity')
        else:
            segments.append('low_activity')
        
        # Risk-based segmentation
        if customer_data.get('inactivity_days', 0) > 30:
            segments.append('high_risk')
        elif customer_data.get('inactivity_days', 0) > 7:
            segments.append('medium_risk')
        else:
            segments.append('low_risk')
        
        return segments
    
    def adjust_risk_thresholds(self, segment, temporal_trend):
        """Dynamically adjust risk thresholds based on segment and trends"""
        base_thresholds = self.risk_thresholds.copy()
        
        # Adjust based on segment
        if 'high_value' in segment:
            # Lower thresholds for high-value customers (more sensitive)
            base_thresholds['medium'] *= 0.8
            base_thresholds['high'] *= 0.9
        elif 'low_value' in segment:
            # Higher thresholds for low-value customers
            base_thresholds['medium'] *= 1.2
            base_thresholds['high'] *= 1.1
        
        # Adjust based on temporal trend
        if temporal_trend == 'declining':
            # Lower thresholds for declining customers
            base_thresholds['medium'] *= 0.9
            base_thresholds['high'] *= 0.85
        elif temporal_trend == 'improving':
            # Raise thresholds for improving customers
            base_thresholds['medium'] *= 1.1
            base_thresholds['high'] *= 1.15
        
        return base_thresholds
    
    def calculate_adaptive_score(self, customer_id, current_data, base_prediction):
        """Calculate adaptive churn score combining multiple factors"""
        
        # Get adaptive features
        adaptive_features = self.calculate_adaptive_features(customer_id, current_data)
        
        # Get customer segments
        segments = self.segment_customers(current_data)
        
        # Get temporal patterns
        temporal = self.analyze_temporal_patterns(customer_id)
        
        # Adjust risk thresholds
        adjusted_thresholds = self.adjust_risk_thresholds(segments, temporal['trend'])
        
        # Calculate component scores
        recent_behavior_score = self._calculate_recent_behavior_score(customer_id)
        historical_pattern_score = self._calculate_historical_pattern_score(customer_id)
        sentiment_trend_score = self._calculate_sentiment_trend_score(customer_id)
        segment_baseline_score = self._calculate_segment_baseline_score(segments)
        
        # Combine scores with adaptive weights
        adaptive_score = (
            self.adaptive_weights['recent_behavior'] * recent_behavior_score +
            self.adaptive_weights['historical_pattern'] * historical_pattern_score +
            self.adaptive_weights['sentiment_trend'] * sentiment_trend_score +
            self.adaptive_weights['segment_baseline'] * segment_baseline_score
        )
        
        # Blend with base ML prediction
        final_score = 0.6 * base_prediction + 0.4 * adaptive_score
        
        # Determine risk level with adjusted thresholds
        if final_score < adjusted_thresholds['low']:
            risk_level = 'Low Risk'
        elif final_score < adjusted_thresholds['medium']:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'
        
        return {
            'adaptive_score': final_score,
            'base_prediction': base_prediction,
            'risk_level': risk_level,
            'adjusted_thresholds': adjusted_thresholds,
            'segments': segments,
            'temporal_patterns': temporal,
            'adaptive_features': adaptive_features,
            'component_scores': {
                'recent_behavior': recent_behavior_score,
                'historical_pattern': historical_pattern_score,
                'sentiment_trend': sentiment_trend_score,
                'segment_baseline': segment_baseline_score
            }
        }
    
    def _calculate_recent_behavior_score(self, customer_id):
        """Calculate score based on recent behavior changes"""
        if customer_id not in self.behavior_history or len(self.behavior_history[customer_id]) < 2:
            return 0.5
        
        history = self.behavior_history[customer_id]
        recent = history[-3:] if len(history) >= 3 else history
        
        if len(recent) < 2:
            return 0.5
        
        # Calculate trend in recent behavior
        login_trend = recent[-1]['login_frequency'] - recent[0]['login_frequency']
        inactivity_trend = recent[-1]['inactivity_days'] - recent[0]['inactivity_days']
        
        # Normalize to 0-1 range (higher is better)
        score = 0.5
        if login_trend > 0:
            score += 0.2
        elif login_trend < -2:
            score -= 0.3
        
        if inactivity_trend < 0:
            score += 0.2
        elif inactivity_trend > 5:
            score -= 0.3
        
        return max(0, min(1, score))
    
    def _calculate_historical_pattern_score(self, customer_id):
        """Calculate score based on historical behavioral patterns"""
        if customer_id not in self.behavior_history or len(self.behavior_history[customer_id]) < 5:
            return 0.5
        
        history = self.behavior_history[customer_id]
        
        # Calculate consistency metrics
        login_freqs = [d['login_frequency'] for d in history]
        inactivity_days = [d['inactivity_days'] for d in history]
        
        # Consistency score (lower variance is better)
        login_consistency = 1 / (1 + np.std(login_freqs))
        inactivity_consistency = 1 / (1 + np.std(inactivity_days))
        
        # Overall pattern score
        pattern_score = (login_consistency + inactivity_consistency) / 2
        
        return max(0, min(1, pattern_score))
    
    def _calculate_sentiment_trend_score(self, customer_id):
        """Calculate score based on sentiment trend"""
        if customer_id not in self.behavior_history or len(self.behavior_history[customer_id]) < 3:
            return 0.5
        
        history = self.behavior_history[customer_id]
        recent_sentiments = [d['sentiment_score'] for d in history[-5:]]
        
        if len(recent_sentiments) < 2:
            return 0.5
        
        # Calculate sentiment trend
        if len(recent_sentiments) >= 4:
            recent_avg = np.mean(recent_sentiments[-2:])
            older_avg = np.mean(recent_sentiments[:-2])
            
            if recent_avg > older_avg + 0.1:
                return 0.8  # Improving sentiment
            elif recent_avg < older_avg - 0.1:
                return 0.2  # Declining sentiment
            else:
                return 0.5  # Stable sentiment
        
        return 0.5
    
    def _calculate_segment_baseline_score(self, segments):
        """Calculate baseline score based on customer segments"""
        score = 0.5
        
        # Value segment adjustments
        if 'high_value' in segments:
            score += 0.1
        elif 'low_value' in segments:
            score -= 0.1
        
        # Activity segment adjustments
        if 'high_activity' in segments:
            score += 0.15
        elif 'low_activity' in segments:
            score -= 0.15
        
        # Risk segment adjustments
        if 'high_risk' in segments:
            score -= 0.2
        elif 'low_risk' in segments:
            score += 0.1
        
        return max(0, min(1, score))
    
    def get_adaptive_insights(self, customer_id, scoring_result):
        """Generate insights and recommendations based on adaptive scoring"""
        insights = []
        
        # Behavioral insights
        if scoring_result['adaptive_features']['login_frequency_change'] < -3:
            insights.append({
                'type': 'warning',
                'title': 'Declining Login Frequency',
                'description': 'Customer login frequency has significantly decreased',
                'recommendation': 'Send re-engagement email with personalized content'
            })
        
        if scoring_result['adaptive_features']['inactivity_change'] > 7:
            insights.append({
                'type': 'critical',
                'title': 'Increasing Inactivity',
                'description': 'Customer inactivity period is growing',
                'recommendation': 'Immediate intervention required - offer incentives'
            })
        
        # Temporal insights
        if scoring_result['temporal_patterns']['trend'] == 'declining':
            insights.append({
                'type': 'warning',
                'title': 'Negative Behavioral Trend',
                'description': 'Customer shows declining behavioral patterns',
                'recommendation': 'Analyze root cause and address pain points'
            })
        
        # Segment-based insights
        if 'high_value' in scoring_result['segments'] and scoring_result['risk_level'] == 'High Risk':
            insights.append({
                'type': 'critical',
                'title': 'High-Value Customer at Risk',
                'description': 'High-value customer showing churn indicators',
                'recommendation': 'Priority intervention with account manager'
            })
        
        # Volatility insights
        if scoring_result['temporal_patterns']['volatility'] > 5:
            insights.append({
                'type': 'info',
                'title': 'Behavioral Volatility',
                'description': 'Customer shows inconsistent usage patterns',
                'recommendation': 'Monitor closely and provide usage guidance'
            })
        
        return insights
    
    def update_adaptive_weights(self, performance_feedback):
        """Update adaptive weights based on model performance feedback"""
        # Simple weight adjustment based on performance
        if performance_feedback.get('recent_behavior_accuracy', 0) > 0.8:
            self.adaptive_weights['recent_behavior'] += 0.05
        if performance_feedback.get('historical_pattern_accuracy', 0) > 0.8:
            self.adaptive_weights['historical_pattern'] += 0.05
        
        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        for key in self.adaptive_weights:
            self.adaptive_weights[key] /= total_weight
