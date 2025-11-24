# 📊 Olist E-commerce Dashboard

## 📝 Description
Interactive e-commerce analytics dashboard built with Streamlit for visualizing and analyzing Olist platform [data](https://www.kaggle.com/code/terencicp/sql-challenge-e-commerce-data-analysis). Provides detailed insights into sales, customers, products, and business performance.

## ✨ Key Features

### 🎯 Advanced Contextual Filters
- **Date Filter**: Date range selection with dual date pickers
- **State Filter**: Multi-select customer states
- **Payment Method Filter**: Multi-select payment types
- **Product Category Filter**: Multi-select categories (recommended max 10)
- **Apply Filters Button**: Applies all filters simultaneously
- **Reset Filters Button**: Resets all filters to default values

### 📈 Key Performance Indicators (KPIs)
- **Revenue**: Total filtered revenue
- **Total Orders**: Number of unique orders
- **Average Order Value**: Average value per order
- **Success Rate**: Percentage of successfully delivered orders
- **Cancellation Rate**: Percentage of cancelled orders
- **Average Rating**: Average customer rating
- **NPS Score**: Net Promoter Score

### 📊 Main Visualizations

#### 1. Payment Distribution
- Interactive bar chart with Plotly
- Shows conversion rates by payment method
- Total values by payment type

#### 2. Lead Sources Distribution
- Interactive pie chart
- Lead source distribution
- Total and per-source lead counters

#### 3. Product Categories Performance
- Stacked bar chart
- Top 10 categories by order volume
- Segmentation by order status

### 🔍 Detailed Analysis (Tabs)

#### Tab 1: Temporal Trends & Forecasting
- Temporal trends analysis
- Prophet forecasting
- Prediction period configuration

#### Tab 2: Delivery Performance
- Delivery time metrics
- Delivery time distribution
- Logistics performance analysis

#### Tab 3: Cohort Analysis
- Customer retention analysis
- Options: monthly/quarterly
- Metrics: customers/revenue
- Interactive heatmap visualization

#### Tab 4: RFM Segmentation
- RFM segmentation (Recency, Frequency, Monetary)
- Customer classification
- Recommended strategies per segment

#### Tab 5: Customer Lifetime Value
- CLV analysis
- Future value predictions
- Customer value segmentation

## 🛠️ Setup and Usage

### Prerequisites
```python
streamlit
pandas
matplotlib
plotly
numpy
prophet (optional for forecasting)
```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

## 📁 Required Data Structure

The dashboard expects the following tables:
- `orders`: Order information
- `order_payments`: Payment details
- `customers`: Customer data
- `products`: Product catalog
- `order_items`: Items per order
- `order_reviews`: Customer reviews
- `product_category_name_translation`: Category translations
- `sellers`: Seller information
- `geolocation`: Geolocation data
- `leads_qualified`: Qualified leads
- `leads_closed`: Closed leads

## 🎨 Customization

### Layout
- **Wide Layout**: Optimized for large screens
- **Sidebar**: Main filters and controls
- **Main Area**: Metrics and visualizations

## 📊 Filtering Options

### Filter Behavior
1. **Cascade**: Filters are applied in cascade
2. **Real-time**: Instant metric updates
3. **Persistence**: Filters persist during session

### Filter Combinations
- All filters can be combined
- Empty filter = All values
- Visual indicators for active filters

## 🔄 Data Flow

1. **Data Loading**: Automatic caching with `@st.cache_data`
2. **Preprocessing**: Date and category normalization
3. **Filter Application**: Dynamic contextual filtering
4. **Metric Calculation**: Real-time updates
5. **Rendering**: Interactive visualizations

## 📝 Important Notes

- **Performance**: For better performance, limit product categories to 10
- **Dates**: System automatically handles timezones
- **Cache**: Data is cached to improve speed
- **Responsive**: Dashboard adapts to different screen sizes

## 🚀 Suggested Future Improvements

- [ ] PDF report export
- [ ] Previous period comparison
- [ ] Automatic alerts
- [ ] Optimized mobile dashboard
- [ ] External API integrations

## 🔧 Configuration Options

### Filter Settings
- **Date Range**: Customizable start and end dates
- **Multi-select Filters**: Hold Ctrl/Cmd for multiple selections
- **Dynamic Updates**: All visualizations update based on filter selections

### Visualization Options
- **Interactive Charts**: Hover for details, click to interact
- **Responsive Design**: Automatically adjusts to screen size
- **Export Options**: Charts can be saved as images

### Performance Tuning
- **Data Caching**: Enabled by default
- **Lazy Loading**: Components load on demand
- **Batch Processing**: Filters apply in batches for efficiency

## 📚 Module Documentation

For questions or issues, review the module documentation:
- `utils.py`: Utility functions
- `key_metrics.py`: Metric calculations
- `cohort_analysis.py`: Cohort analysis
- `rfm_visualization.py`: RFM visualization
- `customer_lifetime_value_visualization.py`: CLV analysis
- `streamlit_prophet_integration.py`: Prophet forecasting integration

## 🐛 Troubleshooting

### Common Issues

1. **Slow Performance**
   - Reduce the number of selected product categories
   - Clear browser cache
   - Restart the Streamlit server

2. **Missing Data**
   - Verify all required tables are present
   - Check date formats in source data
   - Ensure proper column naming

3. **Filter Issues**
   - Click "Reset All Filters" to start fresh
   - Apply filters one at a time to identify issues
   - Check for data availability in selected date range

Built with:
- Streamlit for web framework
- Plotly for interactive visualizations
- Pandas for data manipulation
- Prophet for forecasting capabilities