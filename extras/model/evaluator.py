import json
import pathlib
import pickle
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from scipy import stats
import joblib

# Configuration
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # Fixed path

SALES_COLUMN_SELECTION = [
    'price',
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'sqft_above',
    'sqft_basement',
    'zipcode'
]

OUTPUT_DIR = "model"
PLOTS_DIR = "plots"

class ModelEvaluator:
    """Comprehensive model evaluation and analysis class."""
    
    def __init__(self, suffix: str = ""):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cv_scores = {}
        self.best_params = {}
        self.suffix = f"-{suffix.removeprefix("-")}"

    def load_data(self, sales_path: str, demographics_path: str, 
                  sales_column_selection: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and merge data with improved error handling."""
        print("Loading and preprocessing data...")
        
        # Load sales data
        try:
            sales_data = pd.read_csv(sales_path, usecols=sales_column_selection, 
                                   dtype={'zipcode': str})
            print(f"Loaded {len(sales_data)} records from sales data")
        except FileNotFoundError:
            print(f"Sales data file not found: {sales_path}")
            # Create synthetic data for demonstration
            return self._create_synthetic_data()
        
        # Load demographics data
        try:
            demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})
            print(f"Loaded demographics data with {len(demographics)} records")
        except FileNotFoundError:
            print(f"Demographics file not found: {demographics_path}")
            print("Proceeding with sales data only...")
            y = sales_data.pop('price')
            X = sales_data.drop(columns=['zipcode'], errors='ignore')
            return X, y
        
        # Merge data
        merged_data = sales_data.merge(demographics, how="left", on="zipcode")
        
        # Check for missing values after merge
        missing_demo = merged_data.isnull().sum().sum()
        if missing_demo > 0:
            print(f"Warning: {missing_demo} missing values after demographic merge")
        
        # Remove zipcode and extract target
        merged_data = merged_data.drop(columns="zipcode", errors='ignore')
        y = merged_data.pop('price')
        X = merged_data
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        
        return X, y
    
    def _create_synthetic_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create synthetic housing data for demonstration."""
        print("Creating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.uniform(1, 4, n_samples)
        sqft_living = np.random.normal(2000, 800, n_samples)
        sqft_lot = np.random.normal(8000, 3000, n_samples)
        floors = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples)
        sqft_above = sqft_living * np.random.uniform(0.7, 1.0, n_samples)
        sqft_basement = sqft_living - sqft_above
        
        # Generate realistic price based on features
        price = (
            bedrooms * 50000 +
            bathrooms * 30000 +
            sqft_living * 150 +
            sqft_lot * 5 +
            floors * 20000 +
            np.random.normal(0, 50000, n_samples)
        )
        price = np.maximum(price, 100000)  # Minimum price
        
        X = pd.DataFrame({
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement
        })
        
        y = pd.Series(price, name='price')
        
        return X, y
    
    def perform_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.2, random_state: int = 42):
        """Perform train-test split and store results."""
        self.X_train, self.X_test, self.y_train, self.y_test = \
            model_selection.train_test_split(X, y, test_size=test_size, 
                                           random_state=random_state)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
    
    def hyperparameter_tuning(self, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning for KNN regressor."""
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'kneighborsregressor__n_neighbors': [3, 5, 7, 10, 15, 20, 25],
            'kneighborsregressor__weights': ['uniform', 'distance'],
            'kneighborsregressor__metric': ['euclidean', 'manhattan']
        }

        # Create pipeline
        pipe = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            neighbors.KNeighborsRegressor()
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipe, param_grid, cv=cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")
        
        return grid_search.cv_results_
    
    def evaluate_model_performance(self) -> Dict[str, float]:
        """Comprehensive model performance evaluation."""
        print("Evaluating model performance...")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics_dict = {
            'train_rmse': np.sqrt(metrics.mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(metrics.mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': metrics.mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': metrics.mean_absolute_error(self.y_test, y_test_pred),
            'train_r2': metrics.r2_score(self.y_train, y_train_pred),
            'test_r2': metrics.r2_score(self.y_test, y_test_pred),
            'train_mape': np.mean(np.abs((self.y_train - y_train_pred) / self.y_train)) * 100,
            'test_mape': np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100
        }
        
        # Print results
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Training RMSE:   ${metrics_dict['train_rmse']:,.2f}")
        print(f"Test RMSE:       ${metrics_dict['test_rmse']:,.2f}")
        print(f"Training MAE:    ${metrics_dict['train_mae']:,.2f}")
        print(f"Test MAE:        ${metrics_dict['test_mae']:,.2f}")
        print(f"Training R²:     {metrics_dict['train_r2']:.4f}")
        print(f"Test R²:         {metrics_dict['test_r2']:.4f}")
        print(f"Training MAPE:   {metrics_dict['train_mape']:.2f}%")
        print(f"Test MAPE:       {metrics_dict['test_mape']:.2f}%")
        
        # Overfitting check
        rmse_ratio = metrics_dict['test_rmse'] / metrics_dict['train_rmse']
        r2_diff = metrics_dict['train_r2'] - metrics_dict['test_r2']
        
        print("\n" + "="*50)
        print("OVERFITTING ANALYSIS")
        print("="*50)
        print(f"RMSE Ratio (Test/Train): {rmse_ratio:.3f}")
        print(f"R² Difference (Train-Test): {r2_diff:.4f}")
        
        if rmse_ratio > 1.2 or r2_diff > 0.1:
            print("⚠️  WARNING: Potential overfitting detected!")
        elif rmse_ratio < 0.9:
            print("⚠️  WARNING: Potential underfitting detected!")
        else:
            print("✅ Model appears to have appropriate fit")
        
        return metrics_dict
    
    def cross_validation_analysis(self, cv_folds: int = 5) -> Dict[str, np.ndarray]:
        """Perform cross-validation analysis."""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        # Cross-validation scores
        cv_rmse = np.sqrt(-model_selection.cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=cv_folds, scoring='neg_mean_squared_error'
        ))
        
        cv_r2 = model_selection.cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=cv_folds, scoring='r2'
        )
        
        cv_mae = -model_selection.cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=cv_folds, scoring='neg_mean_absolute_error'
        )
        
        self.cv_scores = {
            'rmse': cv_rmse,
            'r2': cv_r2,
            'mae': cv_mae
        }
        
        print(f"CV RMSE: ${cv_rmse.mean():,.2f} (±${cv_rmse.std()*2:.2f})")
        print(f"CV R²:   {cv_r2.mean():.4f} (±{cv_r2.std()*2:.4f})")
        print(f"CV MAE:  ${cv_mae.mean():,.2f} (±${cv_mae.std()*2:.2f})")
        
        return self.cv_scores
    
    def create_visualizations(self):
        """Create comprehensive visualization plots."""
        print("Creating visualizations...")
        
        # Create plots directory
        plots_dir = pathlib.Path(PLOTS_DIR)
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Learning Curves
        self._plot_learning_curves(plots_dir)
        
        # 2. Validation Curves
        self._plot_validation_curves(plots_dir)
        
        # 3. Residual Analysis
        self._plot_residual_analysis(plots_dir)
        
        # 4. Feature Importance (for tree-based models) or Feature Analysis
        self._plot_feature_analysis(plots_dir)
        
        # 5. Cross-validation results
        self._plot_cv_results(plots_dir)
        
        print(f"Visualizations saved to {plots_dir}")
    
    def _plot_learning_curves(self, plots_dir: pathlib.Path):
        """Plot learning curves to detect overfitting."""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_rmse.mean(axis=1), 'o-', label='Training RMSE')
        plt.fill_between(train_sizes, train_rmse.mean(axis=1) - train_rmse.std(axis=1),
                         train_rmse.mean(axis=1) + train_rmse.std(axis=1), alpha=0.3)
        
        plt.plot(train_sizes, val_rmse.mean(axis=1), 'o-', label='Validation RMSE')
        plt.fill_between(train_sizes, val_rmse.mean(axis=1) - val_rmse.std(axis=1),
                         val_rmse.mean(axis=1) + val_rmse.std(axis=1), alpha=0.3)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f'learning_curves{self.suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_validation_curves(self, plots_dir: pathlib.Path):
        """Plot validation curves for hyperparameter sensitivity."""
        k_range = range(1, 31)
        train_scores, val_scores = validation_curve(
            pipeline.make_pipeline(preprocessing.RobustScaler(), 
                                 neighbors.KNeighborsRegressor()),
            self.X_train, self.y_train, param_name='kneighborsregressor__n_neighbors',
            param_range=k_range, cv=5, scoring='neg_mean_squared_error'
        )
        
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, train_rmse.mean(axis=1), 'o-', label='Training RMSE')
        plt.plot(k_range, val_rmse.mean(axis=1), 'o-', label='Validation RMSE')
        plt.fill_between(k_range, val_rmse.mean(axis=1) - val_rmse.std(axis=1),
                         val_rmse.mean(axis=1) + val_rmse.std(axis=1), alpha=0.3)
        
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('RMSE')
        plt.title('Validation Curve - K-Nearest Neighbors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f'validation_curves{self.suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_analysis(self, plots_dir: pathlib.Path):
        """Create residual analysis plots."""
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residuals vs Predictions
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        
        # 2. QQ plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        
        # 3. Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, density=True, alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # 4. Actual vs Predicted
        axes[1, 1].scatter(self.y_test, y_pred, alpha=0.6)
        axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'red', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted Values')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'residual_analysis{self.suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self, plots_dir: pathlib.Path):
        """Analyze feature distributions and correlations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature correlation heatmap
        corr_matrix = self.X_train.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0, 0], fmt='.2f')
        axes[0, 0].set_title('Feature Correlation Matrix')
        
        # 2. Target distribution
        axes[0, 1].hist(self.y_train, bins=30, density=True, alpha=0.7)
        axes[0, 1].set_xlabel('Price')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Target Variable Distribution')
        
        # 3. Feature importance proxy (correlation with target)
        feature_importance = self.X_train.corrwith(self.y_train).abs().sort_values(ascending=True)
        axes[1, 0].barh(range(len(feature_importance)), feature_importance.values)
        axes[1, 0].set_yticks(range(len(feature_importance)))
        axes[1, 0].set_yticklabels(feature_importance.index)
        axes[1, 0].set_xlabel('Absolute Correlation with Price')
        axes[1, 0].set_title('Feature Importance (Correlation)')
        
        # 4. Price vs most important feature
        most_important_feature = feature_importance.index[-1]
        axes[1, 1].scatter(self.X_train[most_important_feature], self.y_train, alpha=0.6)
        axes[1, 1].set_xlabel(most_important_feature)
        axes[1, 1].set_ylabel('Price')
        axes[1, 1].set_title(f'Price vs {most_important_feature}')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'feature_analysis{self.suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cv_results(self, plots_dir: pathlib.Path):
        """Plot cross-validation results."""
        if not self.cv_scores:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE distribution
        axes[0].boxplot(self.cv_scores['rmse'])
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Cross-Validation RMSE')
        axes[0].grid(True, alpha=0.3)
        
        # R² distribution
        axes[1].boxplot(self.cv_scores['r2'])
        axes[1].set_ylabel('R²')
        axes[1].set_title('Cross-Validation R²')
        axes[1].grid(True, alpha=0.3)
        
        # MAE distribution
        axes[2].boxplot(self.cv_scores['mae'])
        axes[2].set_ylabel('MAE')
        axes[2].set_title('Cross-Validation MAE')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'cv_results{self.suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, metrics: Dict[str, float]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("="*60)
        report.append("MODEL EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Model configuration
        report.append("MODEL CONFIGURATION:")
        report.append(f"- Algorithm: K-Nearest Neighbors Regression")
        report.append(f"- Best parameters: {self.best_params}")
        report.append(f"- Training samples: {len(self.X_train)}")
        report.append(f"- Test samples: {len(self.X_test)}")
        report.append(f"- Features: {len(self.X_train.columns)}")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS:")
        report.append(f"- Test RMSE: ${metrics['test_rmse']:,.2f}")
        report.append(f"- Test MAE: ${metrics['test_mae']:,.2f}")
        report.append(f"- Test R²: {metrics['test_r2']:.4f}")
        report.append(f"- Test MAPE: {metrics['test_mape']:.2f}%")
        report.append("")
        
        # Model fit assessment
        report.append("MODEL FIT ASSESSMENT:")
        rmse_ratio = metrics['test_rmse'] / metrics['train_rmse']
        r2_diff = metrics['train_r2'] - metrics['test_r2']
        
        if rmse_ratio > 1.2 or r2_diff > 0.1:
            report.append("⚠️  OVERFITTING DETECTED")
            report.append("- Test performance significantly worse than training")
            report.append("- Consider: reducing model complexity, more data, regularization")
        elif rmse_ratio < 0.9:
            report.append("⚠️  UNDERFITTING DETECTED") 
            report.append("- Poor performance on both training and test sets")
            report.append("- Consider: increasing model complexity, more features")
        else:
            report.append("✅ APPROPRIATE FIT")
            report.append("- Model generalizes well to unseen data")
        
        report.append("")
        
        # Cross-validation results
        if self.cv_scores:
            cv_rmse_mean = self.cv_scores['rmse'].mean()
            cv_rmse_std = self.cv_scores['rmse'].std()
            report.append("CROSS-VALIDATION RESULTS:")
            report.append(f"- CV RMSE: ${cv_rmse_mean:,.2f} (±${cv_rmse_std*2:.2f})")
            report.append(f"- CV R²: {self.cv_scores['r2'].mean():.4f} (±{self.cv_scores['r2'].std()*2:.4f})")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if metrics['test_r2'] < 0.7:
            report.append("- R² score suggests room for improvement")
            report.append("- Consider feature engineering or different algorithms")
        if metrics['test_mape'] > 20:
            report.append("- High MAPE suggests prediction errors")
            report.append("- Review outliers and data quality")
        
        report.append("- Monitor model performance on new data")
        report.append("- Consider ensemble methods for better predictions")
        report.append("")

        return "\n".join(report)
    
    def save_artifacts(self, save_model: bool):
        """Save model artifacts and evaluation results."""
        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        # Save model
        if save_model:
            pickle.dump(self.model, open(output_dir / f"model{self.suffix}.pkl", 'wb'))

        # Save feature names
        with open(output_dir / f"model_features{self.suffix}.json", 'w') as f:
            json.dump(list(self.X_train.columns), f)

        # Save best parameters
        with open(output_dir / f"best_params{self.suffix}.json", 'w') as f:
            json.dump(self.best_params, f)
        
        # Save cross-validation scores
        if self.cv_scores:
            cv_results = {k: v.tolist() for k, v in self.cv_scores.items()}
            with open(output_dir / f"cv_scores{self.suffix}.json", 'w') as f:
                json.dump(cv_results, f)

        print(f"Model artifacts saved to {output_dir}")


def get_best() -> None:
    """Main execution function with comprehensive evaluation."""
    print("Starting comprehensive model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(suffix="best")

    # Load data
    X, y = evaluator.load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    
    # Train-test split
    evaluator.perform_train_test_split(X, y)
    
    # Hyperparameter tuning
    evaluator.hyperparameter_tuning()
    
    # Evaluate performance
    metrics = evaluator.evaluate_model_performance()
    
    # Cross-validation analysis
    evaluator.cross_validation_analysis()
    
    # Create visualizations
    evaluator.create_visualizations()
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(metrics)
    print("\n" + report)

    # Save artifacts
    evaluator.save_artifacts(save_model=True)

    # Save report
    with open(pathlib.Path(OUTPUT_DIR) / f"evaluation_report{evaluator.suffix}.txt", 'w') as f:
        f.write(report)

    print("\nEvaluation complete! Check the 'model' and 'plots' directories for results.")


def get_existing() -> None:
    # Initialize evaluator
    evaluator = ModelEvaluator(suffix='current')

    # Load data
    X, y = evaluator.load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    
    # Train-test split
    evaluator.perform_train_test_split(X, y)

    with open(pathlib.Path(OUTPUT_DIR) / "model.pkl", "rb") as file:
        evaluator.model = pickle.load(file)
    
    # Evaluate performance
    metrics = evaluator.evaluate_model_performance()
    
    # Cross-validation analysis
    evaluator.cross_validation_analysis()
    
    # Create visualizations
    evaluator.create_visualizations()
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(metrics)
    print("\n" + report)

    evaluator.save_artifacts(save_model=False)

    # Save report
    with open(pathlib.Path(OUTPUT_DIR) / "evaluation_report.txt", 'w') as f:
        f.write(report)


if __name__ == "__main__":
    get_existing()
