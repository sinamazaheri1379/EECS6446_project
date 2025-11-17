import numpy as np
from datetime import datetime, timedelta
import joblib

class ElascalePredictiveEnhancement:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Load pre-trained predictive models"""
        try:
            self.cpu_predictor = joblib.load('models/cpu_predictor.pkl')
            self.traffic_predictor = joblib.load('models/traffic_predictor.pkl')
        except:
            print("Models not found, training new ones...")
            self.train_models()
    
    def enhance_elascale_decision(self, current_metrics, service):
        """Enhance Elascale with predictive capabilities"""
        
        # Get standard Elascale score
        reactive_score = analyze_compute_score_enhanced(current_metrics, service)
        
        # Get predictive score (15-30 min ahead)
        future_metrics = self.predict_future_state(current_metrics, service)
        predictive_score = analyze_compute_score_enhanced(future_metrics, service)
        
        # Combine reactive and predictive
        if service == 'frontend':
            # Frontend needs more proactive scaling
            final_score = max(reactive_score, predictive_score * 1.1)
        elif service == 'cartservice':
            # CartService with aggressive thresholds
            final_score = reactive_score * 0.7 + predictive_score * 0.3
        else:
            final_score = reactive_score * 0.8 + predictive_score * 0.2
            
        return final_score, {
            'reactive': reactive_score,
            'predictive': predictive_score,
            'final': final_score
        }
    
    def predict_future_state(self, current_metrics, service, minutes_ahead=15):
        """Predict metrics 15 minutes ahead"""
        features = self.extract_features(current_metrics)
        
        predicted_cpu = self.cpu_predictor.predict([[
            features['hour'],
            features['minute'],
            features['current_cpu'],
            features['current_users']
        ]])[0]
        
        # Apply service-specific adjustments
        if service == 'cartservice':
            predicted_cpu *= 1.15  # CartService tends to spike
        elif service == 'frontend':
            predicted_cpu *= 1.10  # Frontend buffer
            
        return {
            'cpu_usage': min(predicted_cpu, 1.0),
            'memory_usage': current_metrics['memory_usage'] * 1.05,
            'network_usage': current_metrics['network_usage'],
            'replication_factor': current_metrics['replication_factor']
        }
