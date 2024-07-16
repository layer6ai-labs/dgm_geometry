from data.transforms.unpack import UnpackBatch


class UnpackLID(UnpackBatch):

    def __call__(self, batch):
        """This is an unpacker for the LID datasets, which have the form (x, lid, idx)"""
        return batch[0]
