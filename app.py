from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from functools import wraps
import logging
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 在生产环境中使用安全的密钥

# 配置日志
logging.basicConfig(filename='app.log', level=logging.INFO)

# 请求频率限制装饰器
def limit_requests(f):
    last_requests = {}
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        now = datetime.now()
        if ip in last_requests:
            time_passed = (now - last_requests[ip]).total_seconds()
            if time_passed < 1:  # 限制每秒最多1个请求
                return jsonify({'error': '请求过于频繁，请稍后再试'}), 429
        last_requests[ip] = now
        return f(*args, **kwargs)
    return decorated_function

# 定义FID信号生成函数
def generate_fid_signal(freq, fs, T, A, T2, phi, SNR):
    """生成FID信号"""
    t = np.arange(0, T, 1/fs)
    signal = A * np.exp(-t / T2) * np.cos(2 * np.pi * freq * t + phi)
    noise_power = (A ** 2) / (10 ** (SNR / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(t))
    return t, signal + noise, signal

# 定义迟滞比较器
def hysteresis_comparator(input_signal, high_threshold=None, low_threshold=None, noise_level=None):
    """迟滞比较器实现"""
    if noise_level is None:
        end_segment = input_signal[-int(len(input_signal)*0.1):]
        noise_level = np.std(end_segment)
    
    signal_peak = np.max(np.abs(input_signal[:int(len(input_signal)*0.1)]))
    
    if high_threshold is None:
        high_threshold = noise_level * 3
    if low_threshold is None:
        low_threshold = -high_threshold
    
    max_threshold = signal_peak * 0.3
    high_threshold = min(high_threshold, max_threshold)
    low_threshold = max(low_threshold, -max_threshold)
    
    output = np.zeros_like(input_signal)
    state = 0
    
    for i in range(len(input_signal)):
        if state == 0 and input_signal[i] > high_threshold:
            state = 1
        elif state == 1 and input_signal[i] < low_threshold:
            state = 0
        output[i] = state
    
    return output, high_threshold, low_threshold

# 定义高频时钟计数
def high_freq_counter(edge_times, clock_frequency):
    """模拟超高频时钟计数"""
    if len(edge_times) < 2:
        return [], []
    
    periods = np.diff(edge_times)
    counts = np.round(periods * clock_frequency).astype(np.int64)
    exact_periods = counts / clock_frequency
    
    return counts, exact_periods

# 定义拟合模型函数
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# 定义分析方法
def analyze_fid(freq, fs, T, A, T2, phi, SNR, f_clock, method):
    """分析FID信号，返回频率估计和误差"""
    # 生成信号
    t, s, pure_signal = generate_fid_signal(freq, fs, T, A, T2, phi, SNR)
    
    # 执行迟滞比较器
    square_wave, high_th, low_th = hysteresis_comparator(s)
    
    # 检测上升沿
    rising_edges = np.where(np.diff(square_wave) > 0)[0]
    t_rising = t[rising_edges]
    
    # 执行时钟计数
    clock_counts, _ = high_freq_counter(t_rising, f_clock)
    
    if len(clock_counts) < 3:
        return None, None, None, None, None
    
    # 准备数据
    k = np.arange(1, len(clock_counts) + 1)
    
    # 根据选择的方法进行分析
    if method == "平均法":
        freq_est = f_clock / np.mean(clock_counts)
    elif method == "25%平均法":
        early_n = max(1, len(clock_counts) // 4)
        freq_est = f_clock / np.mean(clock_counts[:early_n])
    elif method == "最小二乘法":
        popt, _ = curve_fit(linear_model, k, clock_counts)
        freq_est = f_clock / linear_model(1, *popt)
    elif method == "二次拟合法":
        popt, _ = curve_fit(quadratic_model, k, clock_counts)
        freq_est = f_clock / quadratic_model(1, *popt)
    elif method == "多项式拟合法":
        weights = np.exp(-k / len(k) * 3)
        poly_coefs = np.polyfit(k, clock_counts, 3, w=weights)
        poly_fit = np.poly1d(poly_coefs)
        freq_est = f_clock / poly_fit(1)
    else:  # 神经网络方法
        # 这里需要实现神经网络方法
        # 暂时返回None
        freq_est = None
    
    if freq_est is not None:
        rel_error = (freq_est - freq) / freq * 100
    else:
        rel_error = None
    
    return freq_est, rel_error, k, clock_counts, (t, s, pure_signal, square_wave)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limit_requests
def analyze():
    try:
        data = request.json
        
        # 记录请求
        logging.info(f"Received analysis request from {request.remote_addr} with parameters: {data}")
        
        # 参数验证
        required_params = ['freq', 'fs', 'T', 'A', 'T2', 'SNR', 'f_clock', 'method']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'缺少参数: {param}'}), 400
        
        # 获取参数
        freq = float(data['freq'])
        fs = float(data['fs'])
        T = float(data['T'])
        A = float(data['A'])
        T2 = float(data['T2'])
        SNR = float(data['SNR'])
        f_clock = float(data['f_clock'])
        method = data['method']
        
        # 分析信号
        freq_est, rel_error, k, clock_counts, signal_data = analyze_fid(
            freq, fs, T, A, T2, 0, SNR, f_clock, method)
        
        if freq_est is None:
            return jsonify({'error': '分析失败'})
        
        # 准备图表数据
        t, s, pure_signal, square_wave = signal_data
        
        # 创建图表数据
        plot_data = {
            'traces': [
                {
                    'x': t.tolist(),
                    'y': s.tolist(),
                    'name': '带噪FID信号',
                    'type': 'scatter',
                    'line': {'color': 'blue'}
                },
                {
                    'x': t.tolist(),
                    'y': pure_signal.tolist(),
                    'name': '无噪声信号',
                    'type': 'scatter',
                    'line': {'color': 'green', 'dash': 'dash'}
                },
                {
                    'x': t.tolist(),
                    'y': (square_wave * np.max(np.abs(s)) * 0.8).tolist(),
                    'name': '方波输出',
                    'type': 'scatter',
                    'line': {'color': 'red'}
                },
                {
                    'x': k.tolist(),
                    'y': clock_counts.tolist(),
                    'name': '时钟计数',
                    'type': 'scatter',
                    'mode': 'markers',
                    'marker': {'color': 'blue'},
                    'xaxis': 'x2',
                    'yaxis': 'y2'
                }
            ],
            'layout': {
                'height': 800,
                'title': 'FID信号分析结果',
                'showlegend': True,
                'grid': {'rows': 2, 'columns': 1},
                'xaxis': {'title': '时间 (s)'},
                'yaxis': {'title': '振幅'},
                'xaxis2': {'title': '周期序号'},
                'yaxis2': {'title': '时钟计数值'}
            }
        }
        
        # 添加拟合曲线
        if method in ["最小二乘法", "二次拟合法", "多项式拟合法"]:
            if method == "最小二乘法":
                popt, _ = curve_fit(linear_model, k, clock_counts)
                fit_func = lambda x: linear_model(x, *popt)
            elif method == "二次拟合法":
                popt, _ = curve_fit(quadratic_model, k, clock_counts)
                fit_func = lambda x: quadratic_model(x, *popt)
            else:  # 多项式拟合法
                weights = np.exp(-k / len(k) * 3)
                poly_coefs = np.polyfit(k, clock_counts, 3, w=weights)
                poly_fit = np.poly1d(poly_coefs)
                fit_func = poly_fit
            
            k_dense = np.linspace(min(k), max(k), 100)
            y_fit = fit_func(k_dense)
            
            plot_data['traces'].append({
                'x': k_dense.tolist(),
                'y': y_fit.tolist(),
                'name': '拟合曲线',
                'type': 'scatter',
                'line': {'color': 'red'},
                'xaxis': 'x2',
                'yaxis': 'y2'
            })
        
        return jsonify({
            'plot_data': plot_data,
            'results': {
                'estimated_frequency': freq_est,
                'relative_error': rel_error,
                'clock_counts': clock_counts.tolist(),
                'parameters': {
                    'freq': freq,
                    'fs': fs,
                    'T': T,
                    'A': A,
                    'T2': T2,
                    'SNR': SNR,
                    'f_clock': f_clock,
                    'method': method
                }
            }
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': '处理请求时发生错误'}), 500

# 添加错误处理
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': '页面未找到'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    # 根据环境变量决定运行模式
    if os.getenv('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
    else:
        app.run(host='0.0.0.0', port=5000, ssl_context='adhoc') 