# 双固化树脂的短程保温核对

配方组把两批双固化树脂混放在同一个样品架上，交接时只留下了四个样品编号和各自的两次短程保温记录。每批树脂都有相同活化能，但没有记录共用指前因子，也没有标出样品属于哪一批。现在需要用八条终点转化率恢复两批样品的归属和各自的初始中间态分数。

程序入口是 `/app/predict.py`。平时由实验数据整理脚本用 `<input.json> <output.json>` 两个位置参数调用。入口可以继续是占位实现，收尾计算仍由镜像里的 Python 完成。

## 实验记录的版式

JSON 顶层恰好包含五个字段。依次是 `schema`、`gas_constant_j_mol_k`、`activation_kj_mol`、`runs`、`fit_tolerance`。`schema` 写作 `"latent-cure-v1"`。气体常数的取值夹在 8.0 与 9.0 之间。`activation_kj_mol` 沿用上一轮质检结果，只接受 40 至 220 kJ/mol 的登记值。`fit_tolerance` 接受 0.000001 至 0.001 的登记值。

`runs` 固定为八条记录。数组元素是带编号和样品号的小对象。对象里还存着温度历程与终点转化率。`id` 形如 `C-[A-Z0-9]{4}` 且全表不重。`sample` 形如 `S-[A-Z0-9]{4}`，四个样品各出现两次。样品编号本身不透露批次。

`profile` 由 3 到 5 个 `[time_s, temperature_k]` 点组成。第一点时间为 0。后续时间严格增加。最后一个时间点不超过 600 s。温度保持在 [300, 450] K。相邻节点之间按线性升降温，节点外不积分。`conversion` 是 [0.01, 0.99] 内的有限数。数组中的布尔值不算数值。

## 固化网络的共同约束

记未反应树脂、潜伏中间态和已固化产物的摩尔分数为 `a`、`x`、`p`。两步反应共用同一个速率常数 `k = A * exp(-E/(R*T))`，其中 `E` 就是输入中的 `activation_kj_mol`，`T` 取当前时刻的线性插值温度。

```text
da/dt = -k * a
dx/dt = k * a - k * x
dp/dt = k * x
```

每一批树脂只有一个初始中间态分数 `initial_intermediate`。初始值为 `a=1-initial_intermediate`、`x=initial_intermediate`、`p=0`。传感器只记录结束时已经进入 `p` 的比例，`x` 始终没有直接测量值。

`initial_intermediate` 的取值范围是 0.02 到 0.48。两个批次的分数相差不少于 0.02。批次号按这个量从小到大排定。`log10_prefactor` 接受 4 至 14 的估计值。

## 交回的诊断结果

拟合把全部记录的终点转化率压到 `fit_tolerance` 的均方根误差以内后，写出一个 JSON 对象。对象包含 `log10_prefactor` 和 `campaigns`。对象还包含 `sample_campaigns` 和 `rms_conversion`。

`log10_prefactor` 是标量。`campaigns` 按 `initial_intermediate` 从小到大排列。`campaigns` 的元素只放一个数，键名是 `initial_intermediate`。`sample_campaigns` 的键必须与输入中的样品集合完全相同。值是 0 或 1。所有数值保留六位小数。

文件写成无空白 JSON。对象键按字母升序。末尾有一个换行。同一份输入再次运行时字节应当一致。

输入字段集合或类型不符时，不生成结果文件。输入数值越出公开范围时，也不生成结果文件。网络无法在容差内闭合时同样不写文件。错误信息写到标准错误。前缀是 `REFUSED: `。退出码为 2。

拟合应当只根据输入内容工作，不能读取固定样例、测试文件或枚举预先知道的组别答案。
