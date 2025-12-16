# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Optional

import modin.pandas as pd
from loguru import logger
from pandas import DataFrame

from experimental.data_feature.dataset_with_swifter import MyDataSetWithSwifter


class MyDataSetWithModin(MyDataSetWithSwifter):

    def preprocess_dispatch(self, lines_data: Optional[DataFrame] = None):
        logger.debug(f"start construct_data with modin, shape {lines_data.shape}")
        try:
            return self.preprocess(pd.DataFrame(lines_data))
        except Exception as e:
            logger.error(f"Failed in construct data with modin. error: {e}")
            return super(MyDataSetWithModin, self).preprocess_dispatch(lines_data)
