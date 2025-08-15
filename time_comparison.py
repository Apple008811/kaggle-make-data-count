#!/usr/bin/env python3
"""
Time Comparison Script
Compares different methods of getting current time
"""

import datetime
import subprocess
import time
import platform

def method1_datetime():
    """Method 1: Python datetime module"""
    try:
        start_time = time.time()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_time = time.time()
        return {
            'method': 'Python datetime',
            'time': current_time,
            'duration': (end_time - start_time) * 1000,  # milliseconds
            'status': 'success'
        }
    except Exception as e:
        return {
            'method': 'Python datetime',
            'time': None,
            'duration': 0,
            'status': f'error: {e}'
        }

def method2_system_command():
    """Method 2: System command (date)"""
    try:
        start_time = time.time()
        
        # Platform-specific command
        if platform.system() == "Windows":
            cmd = ['time', '/t']
        else:
            cmd = ['date']
            
        current_time = subprocess.check_output(cmd).decode().strip()
        end_time = time.time()
        
        return {
            'method': 'System command',
            'time': current_time,
            'duration': (end_time - start_time) * 1000,  # milliseconds
            'status': 'success'
        }
    except Exception as e:
        return {
            'method': 'System command',
            'time': None,
            'duration': 0,
            'status': f'error: {e}'
        }

def method3_os_time():
    """Method 3: OS module"""
    try:
        import os
        start_time = time.time()
        current_time = os.popen('date').read().strip()
        end_time = time.time()
        
        return {
            'method': 'OS module',
            'time': current_time,
            'duration': (end_time - start_time) * 1000,  # milliseconds
            'status': 'success'
        }
    except Exception as e:
        return {
            'method': 'OS module',
            'time': None,
            'duration': 0,
            'status': f'error: {e}'
        }

def main():
    """Main comparison function"""
    print("🕐 TIME GETTING METHODS COMPARISON")
    print("=" * 50)
    
    # System info
    print(f"Platform: {platform.system()}")
    print(f"Python version: {platform.python_version()}")
    print()
    
    # Test all methods
    methods = [method1_datetime, method2_system_command, method3_os_time]
    results = []
    
    for method in methods:
        result = method()
        results.append(result)
        print(f"📋 {result['method']}:")
        print(f"   Time: {result['time']}")
        print(f"   Duration: {result['duration']:.2f} ms")
        print(f"   Status: {result['status']}")
        print()
    
    # Performance comparison
    print("🏆 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['duration'])
        print(f"Fastest method: {fastest['method']} ({fastest['duration']:.2f} ms)")
        
        print("\nSpeed ranking:")
        for i, result in enumerate(sorted(successful_results, key=lambda x: x['duration']), 1):
            print(f"{i}. {result['method']}: {result['duration']:.2f} ms")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("✅ Best choice: Python datetime module")
    print("   - Cross-platform compatible")
    print("   - Fast and reliable")
    print("   - Flexible formatting")
    print("   - No external dependencies")
    
    print("\n❌ Avoid: System commands")
    print("   - Platform dependent")
    print("   - Slower performance")
    print("   - Potential security issues")
    print("   - Error-prone")

if __name__ == "__main__":
    main() 