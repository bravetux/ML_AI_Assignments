"""
3D plot for a house-price prediction model.

This script creates a small synthetic dataset with two features (Size, Age),
fits a linear regression model, and draws a 3D scatter of the true prices and
the predicted surface using matplotlib's 3D plotting.

Outputs saved next to this script:
 - house_price_3d.svg
 - house_price_3d.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used by mpl)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def make_synthetic_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    # Size in square feet (between 500 and 3500)
    size = rng.uniform(500, 3500, size=n)
    # Age in years (0 to 50)
    age = rng.uniform(0, 50, size=n)

    # True underlying model (chosen for illustration)
    # price = 50 * size - 1200 * age + intercept + noise
    intercept = 20000
    price = 50.0 * size - 1200.0 * age + intercept
    # add some noise
    price += rng.normal(0, 20000, size=n)

    df = pd.DataFrame({
        'Size': size,
        'Age': age,
        'Price': price,
    })
    return df


def fit_model(df: pd.DataFrame):
    X = df[['Size', 'Age']].values
    y = df['Price'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred


def plot_3d(df: pd.DataFrame, model, output_dir: Path):
    X = df[['Size', 'Age']].values
    y = df['Price'].values

    # create grid for surface
    size_lin = np.linspace(df['Size'].min(), df['Size'].max(), 40)
    age_lin = np.linspace(df['Age'].min(), df['Age'].max(), 40)
    S, A = np.meshgrid(size_lin, age_lin)
    grid = np.column_stack([S.ravel(), A.ravel()])
    preds = model.predict(grid).reshape(S.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # scatter actual data points
    ax.scatter(df['Size'], df['Age'], df['Price'], color='#2b8cbe', s=40, label='Actual')

    # plot predicted surface
    ax.plot_surface(S, A, preds, cmap='viridis', alpha=0.6, rstride=1, cstride=1, edgecolor='none')

    ax.set_xlabel('Size (sq ft)')
    ax.set_ylabel('Age (years)')
    ax.set_zlabel('Price ($)')
    ax.set_title('House Price vs Size and Age (Linear Regression Surface)')
    ax.view_init(elev=25, azim=-120)

    # metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    # mean_squared_error in older scikit-learn may not support `squared` kwarg
    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    caption = f"R^2 = {r2:.3f}    RMSE = ${rmse:,.0f}"
    fig.text(0.02, 0.02, caption, fontsize=10)

    # save both SVG and PNG next to script
    out_svg = output_dir / 'house_price_3d.svg'
    out_png = output_dir / 'house_price_3d.png'
    fig.savefig(out_svg, bbox_inches='tight', format='svg')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return out_svg, out_png


def main() -> None:
    df = make_synthetic_data(n=80, seed=42)
    model, _ = fit_model(df)

    script_dir = Path(__file__).resolve().parent
    svg_path, png_path = plot_3d(df, model, script_dir)
    print(f"Saved 3D plots to: {svg_path} and {png_path}")


if __name__ == '__main__':
    main()
