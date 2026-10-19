"""业务工具：IT 工单查询、年休假计算。"""
import json

from langchain_core.tools import tool

from agent.config import settings


def _load_tickets() -> dict:
    with open(settings.tickets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {t["id"]: t for t in data.get("tickets", [])}


@tool
def get_ticket_status(ticket_id: str) -> str:
    """根据工单编号查询 IT 服务工单的当前状态、负责人和问题描述。
    当用户提供形如 IT-1001 的工单编号并询问处理进度时使用。"""
    tickets = _load_tickets()
    ticket = tickets.get(ticket_id.strip().upper())
    if not ticket:
        return f"未找到编号为 {ticket_id} 的工单，请核对编号（格式如 IT-1001）。"
    return (
        f"工单编号：{ticket['id']}\n"
        f"问题类型：{ticket['type']}\n"
        f"问题描述：{ticket['desc']}\n"
        f"当前状态：{ticket['status']}\n"
        f"负责人：{ticket['owner']}\n"
        f"最近更新：{ticket['updated']}"
    )


@tool
def calc_annual_leave(years_of_service: int) -> str:
    """根据员工的累计工作年限（整数，单位：年）计算法定带薪年休假天数。
    规则：累计工作满 1 年不满 10 年为 5 天；满 10 年不满 20 年为 10 天；满 20 年为 15 天；不满 1 年为 0 天。"""
    y = int(years_of_service)
    if y < 1:
        days, rule = 0, "累计工作不满 1 年，依法不享受带薪年休假。"
    elif y < 10:
        days, rule = 5, "累计工作满 1 年不满 10 年，年休假 5 天。"
    elif y < 20:
        days, rule = 10, "累计工作满 10 年不满 20 年，年休假 10 天。"
    else:
        days, rule = 20, "累计工作满 20 年，法定15天+公司福利5天，合计20天"
    return f"累计工作年限约 {y} 年：{rule}可休年休假 {days} 天。"
