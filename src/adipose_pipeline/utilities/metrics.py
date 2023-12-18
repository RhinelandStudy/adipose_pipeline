# Copyright 2023 Population Health Sciences and AI in Medical Imaging, German Center for Neurodegenerative Diseases (DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np

def dice(predictions, labels, num_classes):
    """Calculates the categorical Dice similarity coefficients for each class
        between labels and predictions.
    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate the dice
            coefficient for
    Returns:
        np.ndarray: dice coefficient per class
    """
    dice_scores = np.zeros((num_classes))
    for i in range(num_classes):
        tmp_den = (np.sum(predictions == i) + np.sum(labels == i))
        tmp_dice = 2. * np.sum((predictions == i) * (labels == i)) / \
            tmp_den if tmp_den > 0 else 1.
        dice_scores[i] = tmp_dice
    return dice_scores.astype(np.float32)
