import traceback
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import yaml

# from utils.file_utils import read_yaml
def read_yaml(yaml_file, section):
    with open(yaml_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg[section]


# noinspection PyBroadException
class ESClient:
    def __init__(self, config_file="config/es_config.yaml", config_section="es_config"):
        self.config = read_yaml(config_file, config_section)

        # ES配置
        self.hosts = self.config['hosts']
        self.username = self.config['username']
        self.password = self.config['password']
        self.timeout = self.config['timeout']
        self.sniff_on_start = self.config['sniff_on_start']
        self.sniff_on_connection_fail = self.config['sniff_on_connection_fail']
        self.sniff_timeout = self.config['sniff_timeout']
        self.sniffer_timeout = self.config['sniffer_timeout']

        try:
            self.client = Elasticsearch(self.hosts,
                                        http_auth=(self.username, self.password),
                                        timeout=self.timeout,
                                        sniff_on_start=self.sniff_on_start,
                                        sniff_on_connection_fail=self.sniff_on_connection_fail,
                                        sniff_timeout=self.sniff_timeout,
                                        sniffer_timeout=self.sniffer_timeout)
        except Exception:
            traceback.print_exc()
            self.client = None

    def search(self, index, body, source=None):
        """
        查询数据
        :param index:ES中的索引名
        :param body:查询约束条件
        :param source:查询约束条件
        :return:
        """
        # print(index)
        # print(body)
        try:
            return self.client.search(index=index, body=body, _source=source, request_timeout=600)
        except Exception:
            traceback.print_exc()
            return None

    
    def scroll_search(self, index, query, batch_size=1000, active_time='5m', total_size=None):
        """
        使用scroll分批查询数据
        :param index:ES中的索引名
        :param body:查询约束条件
        :param batch_size:每次scroll返回的记录数量
        :param active_time: 连接保持active的时间，默认5分钟
        :param total_size: 查询符合条件的记录数目，若调用该函数时就已知，可以直接传进来
        :return:
        """
        if 'query' not in body:  # 兼容老版本
            body = {'query': body, 'size': batch_size}
        # 查询符合条件的记录数目，用于判断是否需要scroll
        if total_size is None:
            count_res = self.client.count(index=index, body={'query': body['query']})
            total_size = count_res['count']
        if total_size > 10000:  # 需要scroll
            body['size'] = batch_size
            q_res = self.client.search(index=index, body=body, scroll=active_time)
            scroll_id = q_res['_scroll_id']
            scroll_size = len(q_res['hits']['hits'])
            while scroll_size:
                yield q_res
                # 继续scroll
                q_res = self.client.scroll(scroll_id=scroll_id, scroll=active_time)
                scroll_id = q_res['_scroll_id']
                scroll_size = len(q_res['hits']['hits'])
        else:  # 不需要scroll
            if total_size:
                body['size'] = total_size
            q_res = self.client.search(index=index, body=body)
            yield q_res

    def create(self, index, body):
        """
        关键索引
        :param index: ES索引名称
        :param body: 相关设置
        :return:
        """
        try:
            return self.client.indices.create(index=index, body=body)
        except Exception:
            traceback.print_exc()
            return None

    def index(self, index, body, _id=None):
        """
        写入数据
        :param index: ES索引名称
        :param body: 待写入的数据
        :param _id: ID
        :return:
        """
        try:
            if _id:
                self.client.index(index, body, id=_id)
            else:
                self.client.index(index, body)
            return self.refresh(index)
        except Exception:
            traceback.print_exc()
            return None

    def bulk_index(self, index, actions):
        """
        批量写入数据
        :param index: ES索引名称
        :param actions: 待写入的数据
        :return:
        """
        try:
            if len(actions) > 0:
                bulk(self.client, actions, index=index, raise_on_error=True)
                return self.refresh(index)
            return None
        except Exception:
            traceback.print_exc()
            return None


    def delete(self, index, _id):
        """
        根据ID删除数据
        :param index: ES索引名称
        :param _id: 删除ID
        :return:
        """
        try:
            self.client.delete(index, _id)
            return self.refresh(index)
        except Exception:
            traceback.print_exc()
            return None

    def delete_index(self, index):
        """
        删除索引
        :param index:
        :return:
        """
        try:
            self.client.indices.delete(index)
            return self.refresh(index)
        except Exception:
            traceback.print_exc()
            return None

    def delete_by_query(self, index, body):
        """
        根据条件删除数据
        :param index: ES索引名称
        :param body: 删除条件
        :return:
        """
        try:
            self.client.delete_by_query(index, body)
            return self.refresh(index)
        except Exception:
            traceback.print_exc()
            return None


    def delete_by_query_wait_for_completion(self, index, body):
        """
        根据条件删除数据
        :param index: ES索引名称
        :param body: 删除条件
        :return:
        """
        try:
            self.client.delete_by_query(index, body, wait_for_completion=False)
            return self.refresh(index)
        except Exception:
            traceback.print_exc()
            return None

    def exists(self, index, _id):
        """
        根据_id判断数据是否存在
        :param index: ES索引名称
        :param _id: ID
        :return:
        """
        return self.client.exists(index, _id)


    def exists_index(self, index):
        """
        是否存在对应索引
        :param index:
        :return:
        """
        return self.client.indices.exists(index)

    def count(self, index, body):
        """
        按条件计数
        :param index: ES索引名称
        :param body: 查询条件
        :return:
        """
        try:
            count_res = self.client.count(index=index, body=body)
            return count_res['count']
        except Exception:
            traceback.print_exc()
            return None

    def update_by_query(self, index, body):
        """
        更新
        :param index: ES索引名称
        :param body: 更新条件
        :return:
        """
        try:
            self.client.update_by_query(index=index, body=body)
            return self.refresh(index)
        except Exception:
            traceback.print_exc()
            return None

    def update_by_id_and_query(self, index, _id, body):
        try:
            self.client.update(index=index, id=_id, body=body)
            return self.refresh(index)
        except Exception:
            traceback.print_exc()
            return None

    def refresh(self, index):
        """
        刷新索引
        :param index:
        :return:
        """
        try:
            return self.client.indices.refresh(index)
        except Exception:
            traceback.print_exc()
            return None
