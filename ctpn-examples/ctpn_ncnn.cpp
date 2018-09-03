#include "ctpn_ncnn.hpp"

#include <map>
#include <exception>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "connect.h"
#include "nms.h"

CtpnNcnn* CtpnNcnn::ctpn = 0;
int CtpnNcnn::_nThreads = 0;

CtpnNcnn* CtpnNcnn::Get() {
    return ctpn;
}

CtpnNcnn* CtpnNcnn::Get(const string& paramFile, const string& binFile) {
    if(!ctpn) {
        ctpn = new CtpnNcnn(paramFile, binFile);
    }

    return ctpn;
}

CtpnNcnn::CtpnNcnn(const string& paramFile, const string& binFile) {
    this->init(paramFile, binFile);
}

CtpnNcnn::~CtpnNcnn() {
    this->clear();
}

vector<cv::Rect> CtpnNcnn::Forward(const cv::Mat& inputImage) {
    return this->predict(inputImage);
}

bool CtpnNcnn::init(const string& paramFile, const string& binFile) {
    try {
        this->net.load_param(paramFile.c_str());
        this->net.load_model(binFile.c_str());
    }
    catch (exception& e) {
        cout << "exception caught: " << e.what() << endl;
        return false;
    }

    return true;
}

bool CtpnNcnn::clear() {
    try {
        this->net.clear();
    }
    catch (exception& e) {
        cout << "exception caught: " << e.what() << endl;
        return false;
    }

    return true;
}

vector<cv::Rect> CtpnNcnn::predict(const cv::Mat& inputImage) {
    float scaleFactor = computeScaleFactor(inputImage);
    int rows = static_cast<int>(inputImage.rows * scaleFactor);
    int cols = static_cast<int>(inputImage.cols * scaleFactor);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            inputImage.data,
            ncnn::Mat::PIXEL_BGR,
            inputImage.cols, inputImage.rows,
            cols, rows);

    const float mean_vals[3] = { 102.9801f, 115.9465f, 122.7717f };
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = this->net.create_extractor();
    ex.set_light_mode(false);

    ncnn::Mat matScores;
    ncnn::Mat matDeltas;
    ex.input("data", in);
    ex.extract("rpn_cls_prob_reshape", matScores);
    ex.extract("rpn_bbox_pred", matDeltas);

    int scoresWidth = matScores.w;
    int scoresHeight = matScores.h;
    int scoresChannel = matScores.c;
    int deltasWidth = matDeltas.w;
    int deltasHeight = matDeltas.h;
    int deltasChannel = matDeltas.c;

    assert(scoresWidth == deltasWidth);
    assert(scoresHeight == deltasHeight);

    vector<float> scores(scoresHeight*scoresWidth*this->numAnchors);
    vector<float> deltas(scoresHeight*scoresWidth*this->numAnchors*2);

    for(int ic = numAnchors; ic < scoresChannel; ic++) {
        ncnn::Mat score = matScores.channel(ic);
        const float* data = (float*)score.data;
        for(int i = 0; i < scoresHeight*scoresWidth; i++) {
            scores[i*numAnchors + ic-numAnchors] = data[i];
        }
    }

    for(int ic = 0; ic < deltasChannel; ic++) {
        ncnn::Mat delta = matDeltas.channel(ic);
        const float* data = (float*)delta.data;
        for(int i = 0; i < deltasHeight*deltasWidth; i++) {
            deltas[i*deltasChannel + ic] = data[i];
        }
    }

    vector<vector<float> > rois;
    apply_deltas_to_anchors(rois, deltas, scores, 16, scoresHeight, scoresWidth, rows, cols, 0.7);

    vector<vector<float> > textlines;
    this->postprocess(rows, cols, scores, rois, textlines);

    vector<cv::Rect> boxes;
    for(auto textline : textlines) {
        float x1 = textline[0] /= scaleFactor;
        float y1 = textline[1] /= scaleFactor;
        float x2 = textline[2] /= scaleFactor;
        float y2 = textline[3] /= scaleFactor;

        boxes.push_back(cv::Rect(x1, y1, x2-x1+1, y2-y1+1));
    }

    return boxes;
}

float CtpnNcnn::computeScaleFactor(const cv::Mat& inputImage) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    int shorterSide = rows < cols ? rows : cols;
    int longerSide = rows > cols ? rows : cols;

    float scaleFactor = this->minScale / shorterSide;
    if(scaleFactor * longerSide > this->maxScale) {
        scaleFactor = this->maxScale / longerSide;
    }

    return scaleFactor;
}

void CtpnNcnn::postprocess(int rows, int cols, vector<float>& scores,
        vector<vector<float> >& rois, vector<vector<float> >& textlines) {

    // 1. filter low score proposals
	vector<int> keep_idxs;
    for(size_t i = 0; i < scores.size(); i++) {
        if(scores[i] > 0.7) {
            keep_idxs.push_back(i);
        }
    }

	for(size_t i = 0; i < keep_idxs.size(); i++) {
		scores[i] = scores[keep_idxs[i]];
		rois[i] = rois[keep_idxs[i]];
    }
	rois.resize(keep_idxs.size());
	scores.resize(keep_idxs.size());
    
	map<float, int> scores_index;
	for(size_t i = 0; i < scores.size(); i++) {
		scores_index.insert(pair<float,int>(scores[i],i));
    }
	
	vector< vector<float> > roi = rois;
	int i = scores.size()-1;
	for(auto it=scores_index.begin(); it!=scores_index.end(); it++) {
		scores[i] = it->first;
		rois[i] = roi[it->second];
		i--;
	}

    // 2. nms proposals
	keep_idxs = nms_proposals(rois, scores, 0.3);
	for(size_t i = 0; i < keep_idxs.size(); i++) {
		scores[i] = scores[keep_idxs[i]];
		rois[i] = rois[keep_idxs[i]];
	}
	rois.resize(keep_idxs.size());
	scores.resize(keep_idxs.size());

    float maxScore = *std::max_element(scores.begin(), scores.end());
    float minScore = *std::min_element(scores.begin(), scores.end());
    if (maxScore == minScore) {
        for_each(scores.begin(), scores.end(), [&](float& score) { score = 0; });
    }
    else {
        for_each(scores.begin(), scores.end(), [&](float& score) {
                score = (score - minScore) / (maxScore - minScore);
                });
    }

    // 3. connect
    textlines = get_text_lines(rois, scores, rows, cols);

    //filter_boxes
    vector<int> keep_filter = filter_boxes(textlines);
    for(size_t i = 0; i < keep_filter.size(); i++) {
        textlines[i] = textlines[keep_filter[i]];
    }
    textlines.resize(keep_filter.size());

    //nms textlines
    if(textlines.size() != 0) {
        vector<int> keep_indexs = nms_tlines(textlines, 0.3);
        for(size_t i=0; i<keep_indexs.size(); i++) {
            textlines[i] = textlines[keep_indexs[i]];
        }
        textlines.resize(keep_indexs.size());
    }
}

bool apply_deltas_to_anchors(vector< vector<float> >& res, vector<float>& bb_deltas, vector<float>& scores, int stride, int height, int width, int imgh, int imgw, float min_score )
{
	vector<int> heights={11, 16, 23, 33, 48, 68, 97, 139, 198, 283};
	vector<int> widths={16};
	vector<float> score = scores;
	scores.clear();
	int base_size = 16;
	int num_anchors = heights.size()*widths.size();
	//vector< vector<float> > res;
	//locate_anchors: 10x[4,] per feature pixel
	for(int hh=0; hh<height; hh++)
    {
		for(int ww=0; ww<width; ww++)
		{
			// 10 anchors
			int x_ = ww*stride;
			int y_ = hh*stride;
			for(size_t h=0; h<heights.size(); h++)
				for(size_t w=0; w<widths.size(); w++)
				{
					int x1,y1,x2,y2;
					float x_ctr = base_size*0.5;
					float y_ctr = base_size*0.5;
					x1 = x_ctr - widths[w]/2 + x_;
					x2 = x_ctr + widths[w]/2 -1 + x_;
					y1 = y_ctr - heights[h]/2 + y_;
					y2 = y_ctr + heights[h]/2 -1 + y_;
					
					int anchor_y_ctr = (y1+y2)/2;
					int anchor_h = heights[h]+1;
					float delta0 = bb_deltas[2*hh*width*num_anchors+2*ww*num_anchors+2*(h+w)];
					float delta1 = bb_deltas[2*hh*width*num_anchors+2*ww*num_anchors+2*(h+w)+1];
					float global_coords1 = anchor_h*exp(delta1);
					float global_coords0 = delta0*anchor_h+anchor_y_ctr-global_coords1/2;
					if(score[hh*width*num_anchors+ww*num_anchors+(h+w)]<min_score)
						continue;
					scores.push_back(score[hh*width*num_anchors+ww*num_anchors+(h+w)]);
					vector<float> rect;
					rect.push_back(threshold(x1,0,imgw-1));
					rect.push_back(threshold(global_coords0,0,imgh-1));
					rect.push_back(threshold(x2,0,imgw-1));
					rect.push_back(threshold(global_coords0+global_coords1,0,imgh-1));
					res.push_back(rect);
				}
		}
    }

	return true;
}
