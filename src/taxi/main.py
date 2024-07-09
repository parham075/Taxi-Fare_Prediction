# pipelines
from taxi.pipelines.pipeline_QA import PiPelineQA
from taxi.pipelines.pipeline_preprocessing import PipelinePreprocessing
from taxi.pipelines.pipeline_train_and_evaluation import PipelineTrainAndEvaluation
from loguru import logger


def main():
    step_name = "QA and QA.1"
    print(
        "************************\n************************\n************************"
    )
    logger.info(f"step_name:  {step_name}")
    data_obj = PiPelineQA()
    ####################################
    ####################################
    step_name = "preprocessing"
    print(
        "************************\n************************\n************************"
    )
    logger.info(f"step_name:  {step_name}")
    X_train, y_train, X_test, y_test = PipelinePreprocessing(data_obj)
    logger.info("Data is preprocessed")
    ########################################
    ###################################
    step_name = "Train and Evaluate"
    print(
        "************************\n************************\n************************"
    )
    logger.info(f"step_name:  {step_name}")
    PipelineTrainAndEvaluation(X_train, y_train, X_test, y_test)
    logger.info("Model is trained and evaluated")


if __name__ == "__main__":
    # Log the experiment
    main()
