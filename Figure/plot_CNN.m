
clear; clc; close all;
%储存位置
save_dir = './figures';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

n_subjects = 16;%被试数量

target_window = 1;%B类的窗口

results(1).filepath = '../Initial/CNN-baseline_loso/1s/result_loso.csv';
results(1).type     = 'A';
results(1).method   = 'LOSO';
results(1).window   = 1;

results(2).filepath = '../Initial/CNN-baseline_loso/2s/result_loso.csv';
results(2).type     = 'A';
results(2).method   = 'LOSO';
results(2).window   = 2;

results(3).filepath = '../Initial/CNN-baseline_loso/5s/result_loso.csv';
results(3).type     = 'A';
results(3).method   = 'LOSO';
results(3).window   = 5;

results(4).filepath = '../Initial/CNN-baseline_loso/10s/result_loso.csv';
results(4).type     = 'A';
results(4).method   = 'LOSO';
results(4).window   = 10;

results(5).filepath = '../Initial/CNN-baseline_loto/1s/result_loto.csv';
results(5).type     = 'B';
results(5).method   = 'LOTO';
results(5).window   = 1;

results(6).filepath = '../Initial/CNN-baseline_loto/2s/result_loto.csv';
results(6).type     = 'B';
results(6).method   = 'LOTO';
results(6).window   = 2;

results(7).filepath = '../Initial/CNN-baseline_loto/5s/result_loto.csv';
results(7).type     = 'B';
results(7).method   = 'LOTO';
results(7).window   = 5;

results(8).filepath = '../Initial/CNN-baseline_loto/10s/result_loto.csv';
results(8).type     = 'B';
results(8).method   = 'LOTO';
results(8).window   = 10;

results(9).filepath  = '../Initial/CNN-baseline_Subject_adaptive/1s/result_subject_adaptive.csv';
results(9).type      = 'A';
results(9).method    = 'Subject-Adaptive';
results(9).window    = 1;

results(10).filepath = '../Initial/CNN-baseline_Subject_adaptive/2s/result_subject_adaptive.csv';
results(10).type     = 'A';
results(10).method   = 'Subject-Adaptive';
results(10).window   = 2;

results(11).filepath = '../Initial/CNN-baseline_Subject_adaptive/5s/result_subject_adaptive.csv';
results(11).type     = 'A';
results(11).method   = 'Subject-Adaptive';
results(11).window   = 5;

results(12).filepath = '../Initial/CNN-baseline_Subject_adaptive/10s/result_subject_adaptive.csv';
results(12).type     = 'A';
results(12).method   = 'Subject-Adaptive';
results(12).window   = 10;

results(13).filepath = '../Initial/CNN-baseline_within_trial/1s/result_within_trial_subject_level_5folds.csv';
results(13).type     = 'B';
results(13).method   = 'Within-Subject 5-Fold';
results(13).window   = 1;

results(14).filepath = '../Initial/CNN-baseline_within_trial/2s/result_within_trial_subject_level_5folds.csv';
results(14).type     = 'B';
results(14).method   = 'Within-Subject 5-Fold';
results(14).window   = 2;

results(15).filepath = '../Initial/CNN-baseline_within_trial/5s/result_within_trial_subject_level_5folds.csv';
results(15).type     = 'B';
results(15).method   = 'Within-Subject 5-Fold';
results(15).window   = 5;

results(16).filepath = '../Initial/CNN-baseline_within_trial/10s/result_within_trial_subject_level_5folds.csv';
results(16).type     = 'B';
results(16).method   = 'Within-Subject 5-Fold';
results(16).window   = 10;

% 颜色
colors = [
    0.12 0.47 0.71;   % 蓝
    0.89 0.29 0.20;   % 红
    0.30 0.69 0.31;   % 绿
    0.60 0.31 0.64    % 紫
];

% 数据读取
fprintf('读取数据中...\n');
num_files = length(results);

% 获取所有方法名和窗口值
method_names = unique({results.method}, 'stable');
num_methods  = length(method_names);
window_vals  = unique([results.window]);
num_windows  = length(window_vals);

% 检查方法数是否等于颜色数
if num_methods ~= size(colors, 1)
    error('颜色数量与划分方法数量不匹配，请检查 colors 矩阵');
end

% 存储每个文件的被试平均准确率（16x1）
all_subject_acc = cell(num_files, 1);

for idx = 1:num_files
    raw = readmatrix(results(idx).filepath);
    if size(raw, 1) ~= n_subjects
        error('文件 %s 的行数不是 %d', results(idx).filepath, n_subjects);
    end
    if strcmp(results(idx).type, 'A')
        acc = raw(:, 1);
    elseif strcmp(results(idx).type, 'B')
        acc = mean(raw, 2);
    else
        error('未知类型：%s', results(idx).type);
    end
    all_subject_acc{idx} = acc;
end

% 重组为：method_acc_mean(method_idx, window_idx) 和 method_acc_std
method_acc_mean = zeros(num_methods, num_windows);
method_acc_std  = zeros(num_methods, num_windows);

for mi = 1:num_methods
    for wi = 1:num_windows
        idx = find(strcmp({results.method}, method_names{mi}) & ...
                   [results.window] == window_vals(wi));
        if isempty(idx)
            warning('缺少方法 %s 窗口 %d 的数据', method_names{mi}, window_vals(wi));
            method_acc_mean(mi, wi) = NaN;
            method_acc_std(mi, wi) = NaN;
        else
            acc_vec = all_subject_acc{idx};
            method_acc_mean(mi, wi) = mean(acc_vec);
            method_acc_std(mi, wi) = std(acc_vec);
        end
    end
end

% 提取 target_window 下的数据用于图二
target_acc = zeros(n_subjects, num_methods);
found_target = false;
for mi = 1:num_methods
    idx = find(strcmp({results.method}, method_names{mi}) & ...
               [results.window] == target_window);
    if ~isempty(idx)
        target_acc(:, mi) = all_subject_acc{idx};
        found_target = true;
    else
        warning('目标窗口 %d 下没有 %s 的数据', target_window, method_names{mi});
        target_acc(:, mi) = NaN;
    end
end
if ~found_target
    error('在目标窗口 %d 下没有找到任何数据', target_window);
end



% ============ 图一 ============
% 单独图
for mi = 1:num_methods
    figure('Position', [100, 100, 700, 500]);
    
    plot(window_vals, method_acc_mean(mi, :), '-o', ...
         'Color', colors(mi, :), 'LineWidth', 2.5, ...
         'MarkerSize', 8, 'MarkerFaceColor', colors(mi, :));
    
    xlabel('Decision Window (s)', 'FontSize', 12);
    ylabel('Accuracy (%)', 'FontSize', 12);
    title(['Fig1 - ', method_names{mi}], 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([min(window_vals)-0.5, max(window_vals)+0.5]);
    ylim([0 100]);
    set(gca, 'XTick', window_vals);
    
    % 保存
    fname = ['Fig1_', strrep(method_names{mi}, ' ', '_')];
    saveas(gcf, fullfile(save_dir, [fname, '.png']));
    saveas(gcf, fullfile(save_dir, [fname, '.fig']));
    fprintf('Saved: %s\n', fname);
end

%综合图
figure('Position', [100, 100, 750, 520]);
hold on;
for mi = 1:num_methods
    plot(window_vals, method_acc_mean(mi, :), '-o', ...
         'Color', colors(mi, :), 'LineWidth', 2.2, ...
         'MarkerSize', 7, 'MarkerFaceColor', colors(mi, :), ...
         'DisplayName', method_names{mi});
end
hold off;

xlabel('Decision Window (s)', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title('Fig1-Combined: Accuracy vs Decision Window', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([min(window_vals)-0.5, max(window_vals)+0.5]);
ylim([0 100]);
set(gca, 'XTick', window_vals);

fname = 'Fig1_Combined';
saveas(gcf, fullfile(save_dir, [fname, '.png']));
saveas(gcf, fullfile(save_dir, [fname, '.fig']));
fprintf('Saved: %s\n', fname);

% 图二
methods_for_fig2A = {'LOSO', 'LOTO', 'Subject-Adaptive'};
methods_for_fig2B = {'Within-Subject 5-Fold'}; 

idx_A = find(ismember(method_names, methods_for_fig2A));
idx_B = find(ismember(method_names, methods_for_fig2B));

figure('Position', [100, 100, 1400, 500]);
num_A = length(idx_A);
bar_width = 0.2;
group_offset = linspace(-1.2*bar_width, 1.2*bar_width, num_A);  % 三个方法

hold on;
for i = 1:num_A
    mi = idx_A(i);
    x_pos = (1:n_subjects) + group_offset(i);
    bar(x_pos, target_acc(:, mi), bar_width, ...
        'FaceColor', colors(mi, :), 'EdgeColor', 'k', 'LineWidth', 0.5, ...
        'DisplayName', method_names{mi});
end

subject_mean_A = mean(target_acc(:, idx_A), 2, 'omitnan');
plot(1:n_subjects, subject_mean_A, '-s', ...
    'Color', 'k', 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'white', ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Subject Mean');

xlabel('Subject ID', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title(sprintf('Fig2A: Per-Subject Accuracy (Window = %ds, LOSO/LOTO/SA)', target_window), ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 10);
xlim([0.2, n_subjects + 0.8]);
ylim([0 105]);
grid on;
set(gca, 'XTick', 1:n_subjects, 'FontSize', 9);

fname = sprintf('Fig2A_Window%ds', target_window);
saveas(gcf, fullfile(save_dir, [fname, '.png']));
saveas(gcf, fullfile(save_dir, [fname, '.fig']));
fprintf('Saved: %s\n', fname);

figure('Position', [100, 100, 1200, 450]);
mi_B = idx_B(1);  % 只有一个方法
bar(1:n_subjects, target_acc(:, mi_B), 'FaceColor', colors(mi_B, :), ...
    'EdgeColor', 'k', 'LineWidth', 0.8);
xlabel('Subject ID', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title(sprintf('Fig2B: Per-Subject Accuracy (Window = %ds, %s)', target_window, method_names{mi_B}), ...
      'FontSize', 14, 'FontWeight', 'bold');
xlim([0.3, n_subjects + 0.7]);
ylim([0 105]);
grid on;
set(gca, 'XTick', 1:n_subjects, 'FontSize', 10);

fname = sprintf('Fig2B_Window%ds', target_window);
saveas(gcf, fullfile(save_dir, [fname, '.png']));
saveas(gcf, fullfile(save_dir, [fname, '.fig']));
fprintf('Saved: %s\n', fname);