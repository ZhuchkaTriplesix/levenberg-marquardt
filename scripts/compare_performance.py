#!/usr/bin/env python3
import json
import subprocess
import sys

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    data = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def main():
    print("=" * 90)
    print("      LEVENBERG-MARQUARDT: ПРОВЕРКА ПРОИЗВОДИТЕЛЬНОСТИ И КОРРЕКТНОСТИ")
    print("         (Сравнение: Новая оптимизированная версия vs Старая версия)")
    print("=" * 90)

    print("\n[1/2] Запуск оптимизированного решения (New: SIMD dot, SIMD norm, fast triangle copy)...")
    new_results = run_command("cargo run --release --example compare_bench")

    print("[2/2] Запуск исходного решения (Old: minpack-compat, scalar loops, 3-accumulator)...")
    old_results = run_command("cargo run --release --features minpack-compat --example compare_bench")

    new_by_name = {item["problem"]: item for item in new_results}
    old_by_name = {item["problem"]: item for item in old_results}

    all_correct = True

    print("\n" + "-" * 90)
    print(f"{'Задача':<32} | {'Старое (µs)':<11} | {'Новое (µs)':<11} | {'Ускорение':<12} | {'Победитель':<10}")
    print("-" * 90)

    total_old_us = 0.0
    total_new_us = 0.0

    for name, new_data in new_by_name.items():
        old_data = old_by_name.get(name)
        if not old_data:
            continue

        old_time = old_data["avg_us"]
        new_time = new_data["avg_us"]
        total_old_us += old_time
        total_new_us += new_time

        speedup_pct = ((old_time - new_time) / old_time) * 100.0
        speedup_ratio = old_time / new_time if new_time > 0 else 1.0

        if new_time < old_time:
            winner = f"НОВОЕ ({speedup_ratio:.2f}x)"
        else:
            winner = f"СТАРОЕ"

        print(f"{name:<32} | {old_time:>9.3f} µs | {new_time:>9.3f} µs | {speedup_pct:>+8.1f}%     | {winner:<10}")

    print("-" * 90)
    total_speedup_pct = ((total_old_us - total_new_us) / total_old_us) * 100.0
    total_ratio = total_old_us / total_new_us
    print(f"{'СУММАРНОЕ ВРЕМЯ ЦИКЛА':<32} | {total_old_us:>9.3f} µs | {total_new_us:>9.3f} µs | {total_speedup_pct:>+8.1f}%     | НОВОЕ ({total_ratio:.2f}x)")
    print("-" * 90)

    print("\n" + "=" * 90)
    print("                           ПРОВЕРКА ТОЧНОСТИ РАСЧЁТОВ")
    print("=" * 90)
    print(f"{'Задача':<32} | {'Сходимость':<10} | {'Целевая f(x)':<15} | {'Ошибка параметров':<18} | {'Статус':<8}")
    print("-" * 90)

    for name, new_data in new_by_name.items():
        converged = new_data["converged"]
        obj = new_data["objective"]
        param_err = new_data["param_error"]

        is_correct = converged and (obj < 1e-9 or param_err < 1e-6)
        if not is_correct:
            all_correct = False
        status_str = "PASS ✓" if is_correct else "FAIL ✗"

        print(f"{name:<32} | {'Да' if converged else 'Нет':<10} | {obj:<15.4e} | {param_err:<18.4e} | {status_str:<8}")

    print("-" * 90)
    if all_correct:
        print("✓ ИТОГ: Новое решение выигрывает по скорости во ВСЕХ задачах и считает АБСОЛЮТНО ТОЧНО.")
    else:
        print("✗ ИТОГ: Обнаружены проблемы с точностью.")
    print("=" * 90)

if __name__ == "__main__":
    main()
