function handleimage(){
    const image = this.files;
    console.log(image);
    Notiflix.Report.info(
                '选择图片',
                '项目负责人：陆珉俊；对话维护：侯吟丞；',
                '确定'
            );
}