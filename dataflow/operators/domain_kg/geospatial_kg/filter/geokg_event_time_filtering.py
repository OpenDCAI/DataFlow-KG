import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import re
from typing import List, Optional
from datetime import datetime, timedelta

@OPERATOR_REGISTRY.register()
class GeoKGEventTupleTimeFilter(OperatorABC):
    """
    Filter spatio-temporal event tuples based on time constraints.

    Event tuple format:
        "<event> ... <location> ... <time> ... <optional fields>"
    """

    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGEventTupleTimeFilter 根据时间条件筛选事件多元组。",
                "输入: List[str]\n输出: List[str]"
            )
        return (
            "KGEventTupleTimeFilter filters spatio-temporal event tuples by time constraints.",
            "Input: List[str]\nOutput: List[str]"
        )

    # ------------------- 时间解析方法 -------------------
    def _extract_time(self, tuple_str: str) -> Optional[str]:
        m = re.search(r"<time>\s*(.*?)(\s*<|$)", tuple_str)
        if not m:
            return None
        return m.group(1).strip()

    def _parse_date(self, t: str):
        if not t or t == "NA":
            return None
        formats = ["%Y-%m-%d","%B %Y","%b %Y","%Y"]
        for f in formats:
            try:
                return datetime.strptime(t,f)
            except:
                pass
        return None

    def _parse_quarter(self,t):
        m=re.match(r"Q([1-4])\s+(\d{4})",t)
        if not m:
            return None
        q=int(m.group(1))
        year=int(m.group(2))
        start_month=(q-1)*3+1
        start=datetime(year,start_month,1)
        end=datetime(year,start_month+2,28)
        return start,end

    def _parse_month(self,t):
        try:
            dt=datetime.strptime(t,"%B %Y")
        except:
            try:
                dt=datetime.strptime(t,"%b %Y")
            except:
                return None
        year,month=dt.year,dt.month
        start=datetime(year,month,1)
        if month==12:
            end=datetime(year,12,31)
        else:
            end=datetime(year,month+1,1)-timedelta(days=1)
        return start,end

    def _parse_tuple_time(self,t):
        if t=="NA" or not t:
            return None,None
        if "|" in t:
            parts=t.split("|")
            try:
                return datetime.strptime(parts[0],"%Y-%m-%d"),datetime.strptime(parts[1],"%Y-%m-%d")
            except:
                return None,None
        q=self._parse_quarter(t)
        if q:
            return q
        m=self._parse_month(t)
        if m:
            return m
        d=self._parse_date(t)
        if d:
            return d,d
        return None,None

    def _parse_query_time(self,t):
        if not t:
            return None,None
        q=self._parse_quarter(t)
        if q:
            return q
        m=self._parse_month(t)
        if m:
            return m
        d=self._parse_date(t)
        if d:
            return d,d
        return None,None

    def _match_time(self,tuple_start,tuple_end,query_start,query_end):
        if tuple_start is None:
            return False
        if query_start and not query_end:
            return tuple_end>=query_start
        if query_start and query_end:
            return not(tuple_end<query_start or tuple_start>query_end)
        return False

    # ------------------- 核心过滤 -------------------
    def _filter(self,tuples:List[str],query_start,query_end)->List[str]:
        results=[]
        for t in tuples:
            tuple_time=self._extract_time(t)
            if not tuple_time:
                continue
            start,end=self._parse_tuple_time(tuple_time)
            if self._match_time(start,end,query_start,query_end):
                results.append(t)
        return results

    # ------------------- DataFrame 验证 -------------------
    def _validate_dataframe(self,dataframe:pd.DataFrame,input_key):
        if input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {input_key}")

    # ------------------- 主运行方法 -------------------
    def run(
        self,
        storage:DataFlowStorage,
        input_key:str="tuple",
        output_key:str="filtered_tuple",
        query_time_start:str="Q1 2021",
        query_time_end:str="2023",
    ):
        self.input_key=input_key
        self.output_key=output_key
        df=storage.read("dataframe")
        self._validate_dataframe(df,input_key)
        self.logger.info("Filtering event tuples by temporal constraints")

        query_start,_=self._parse_query_time(query_time_start)
        query_end=None
        if query_time_end:
            _,query_end=self._parse_query_time(query_time_end)

        results=[]
        for row in df[input_key]:
            if not isinstance(row,list):
                results.append([])
                continue
            filtered=self._filter(row,query_start,query_end)
            results.append(filtered)

        if self.merge_to_input:
            df[self.input_key]=results
            output_file=storage.write(df)
            self.logger.info(f"Results saved to {output_file}")
            return [self.input_key]

        df[self.output_key]=results
        output_file=storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]