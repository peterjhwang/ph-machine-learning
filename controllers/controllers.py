from flask_app import application
from flask import jsonify
from services.data_processing.data_preparation import data_preparation
from services.data_processing.feature_engineering import feature_engineering

@application.route('/prepare')
def prepare():
    try:
        data_preparation()
        return jsonify({'message': 'Data loaded into S3'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@application.route('/create_features')
def create_features():
    try:
        feature_engineering()
        return jsonify({'message': 'Feature added data loaded into S3'})
    except Exception as e:
        return jsonify({'error': str(e)})