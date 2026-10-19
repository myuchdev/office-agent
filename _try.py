# -*- coding: utf-8 -*-
r"""练习沙盒：想快速测试某个工具时，把测试代码写在这里，然后运行：
    .\.venv\Scripts\python.exe _try.py
本文件不影响项目，可随时改、用完可删。"""

from agent.tools.business import calc_annual_leave

# 分别测试：不满1年、满1年、正好满10年、满20年（你改过的福利档）
for y in [0, 5, 10, 22]:
    print(calc_annual_leave.invoke({"years_of_service": y}))
