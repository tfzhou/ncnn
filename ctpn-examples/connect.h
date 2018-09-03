#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

#define MAX_HORIZONTAL_GAP 50
#define MIN_V_OVERLAPS 0.7
#define MIN_SIZE_SIM 0.7
#define MIN_RATIO 1.2
#define LINE_MIN_SCORE 0.7
#define TEXT_PROPOSALS_WIDTH 16
#define MIN_NUM_PROPOSALS 2

float overlaps_v(int index1, int index2, vector< vector<float> >& text_proposals)
{
    float h1 = text_proposals[index1][3] - text_proposals[index1][1] + 1;
    float h2 = text_proposals[index2][3] - text_proposals[index2][1] + 1;
    float y0 = max(text_proposals[index1][1], text_proposals[index2][1]);
    float y1 = min(text_proposals[index1][3], text_proposals[index2][3]);
    return max(float(0), y1-y0+1)/min(h1, h2);
}


float size_similarity(int index1, int index2, vector< vector<float> >& text_proposals)
{
    float h1 = text_proposals[index1][3] - text_proposals[index1][1] + 1;
    float h2 = text_proposals[index2][3] - text_proposals[index2][1] + 1;
    return min(h1, h2)/max(h1, h2);
}


int meet_v_iou(int index1, int index2, vector< vector<float> >& text_proposals)
{
    return (overlaps_v(index1, index2, text_proposals) >= MIN_V_OVERLAPS) && (size_similarity(index1, index2, text_proposals) >= MIN_SIZE_SIM);
}


vector<int> get_sucessions(vector< vector<float> >& text_proposals, int index, int w, vector< vector<int> >& boxes_table)
{
    vector<int> results;
    vector<float> box = text_proposals[index];
    for(size_t left=box[0]+1; left<min(box[0]+MAX_HORIZONTAL_GAP+1, float(w)); left++){
        vector<int> adj_box_index = boxes_table[left];
        for(size_t i=0; i<adj_box_index.size(); i++){
            if(meet_v_iou(adj_box_index[i], index, text_proposals)==1){
                results.push_back(adj_box_index[i]);
            }
        }
        if(results.size()!=0){
            return results;
        }
    }
    return results;
}


vector<int> get_precursors(vector< vector<float> >& text_proposals, int index, vector< vector<int> >& boxes_table)
{
    vector<int> results;
    vector<float> box = text_proposals[index];
    for(int left=box[0]-1; left>max(box[0]-MAX_HORIZONTAL_GAP, float(0))-1; left--){
        vector<int> adj_box_index = boxes_table[left];
        for(size_t i=0; i<adj_box_index.size(); i++){
            if(meet_v_iou(adj_box_index[i], index, text_proposals)==1){
                results.push_back(adj_box_index[i]);
            }
        }
        if(results.size()!=0){
            return results;
        }
    }
    return results;
}


bool is_succession_node(int index, int succession_index, vector<float>& scores, vector< vector<float> >& text_proposals, vector< vector<int> >& boxes_table)
{
    vector<int> precursors = get_precursors(text_proposals, succession_index, boxes_table);

    int  precursor_index =precursors[0];
    for(size_t j=0; j<precursors.size(); j++){
        if(scores[precursors[j]]>scores[precursor_index]){
            precursor_index = precursors[j];
        }
    }
    if(scores[index] >= scores[precursor_index]){
        return true;
    }
    return false;
}


vector< vector<int> > build_graph(vector< vector<float> >& text_proposals, vector<float>& scores, int h, int w)
{
   vector< vector<int> > boxes_table(w);
   for(size_t i=0; i<text_proposals.size(); i++){
       boxes_table[text_proposals[i][0]].push_back(i);
   }
   /*for(int i=0; i<w; i++){
       for(int j=0; j<boxes_table[i].size(); j++){
           cout<<boxes_table[i][j]<<" ";
       }
       cout<<endl;
   }*/

   vector< vector<int> > graph(text_proposals.size() ,vector<int>(text_proposals.size(), 0));

   for(size_t i=0; i<text_proposals.size(); i++){
       vector<int> successions;
       successions = get_sucessions(text_proposals, i, w, boxes_table);
       if(successions.size() == 0)
           continue;
       int succession_index = successions[0];
       for(size_t j=0; j<successions.size(); j++){
           if(scores[successions[j]]>scores[succession_index]){
               succession_index = successions[j];
           }
       }
       if(is_succession_node(i, succession_index, scores, text_proposals, boxes_table)){
           graph[i][succession_index] = 1;
       }
   }
   return graph;
}


vector< vector<int> > sub_graphs_connected(vector< vector<int> > graph)
{
    vector< vector<int> > sub_graphs;
    for(size_t i=0; i<graph[0].size(); i++){
        size_t j;
        for(j=0; j<graph[0].size(); j++){
            if(graph[j][i] == 1){
                break;
            }
        }
        if(i<graph[0].size() && j>=graph[0].size()){
            vector<int> tmp;
            tmp.push_back(i);
            //cout<<i<<" ";
            int t = i;
            while(1){
                size_t k;
                for(k=0; k<graph[0].size(); k++){
                    if(graph[t][k] == 1){
                        tmp.push_back(k);
                        //cout<<k<<" "; 
                        t = k;
                        break;
                    }
                }
                if(k >= graph[0].size()){
                    break;
                }
            }
            sub_graphs.push_back(tmp);
            //cout<<endl;
        }
    }
    return sub_graphs;
}


vector< vector<int> > group_text_proposals(vector< vector<float> >& text_proposals, vector<float>& scores, int h, int w)
{
    vector< vector<int> > graph = build_graph(text_proposals, scores, h, w);
    /*for(int i=0; i<graph.size(); i++){
        for(int j=0; j<graph[i].size(); j++){
            cout<<graph[i][j]<<" ";
        }
        cout<<endl;
    }*/
    return sub_graphs_connected(graph);
}


float fit_y1(vector< vector<float> >& text_line_boxes, float x)
{
    if(text_line_boxes.size()==1){
        return text_line_boxes[0][1];
    }
    float a=0, b=0, c=0, d=0;
    int n = text_line_boxes.size();
    for(int i=0; i<n; i++){
        a += text_line_boxes[i][0]*text_line_boxes[i][0];
        b += text_line_boxes[i][0];
        c += text_line_boxes[i][0]*text_line_boxes[i][1];
        d += text_line_boxes[i][1];
    }
    float k = (c*n-b*d)/(a*n-b*b);
    float bb = (a*d-c*b)/(a*n-b*b);
    //cout<<"1 "<<k<<" "<<bb<<endl;
    float p = k*x + bb;
    return p;
}


float fit_y2(vector< vector<float> >& text_line_boxes, float x)
{
    if(text_line_boxes.size()==1){
        return text_line_boxes[0][1];
    }
    float a=0, b=0, c=0, d=0;
    int n = text_line_boxes.size();
    for(int i=0; i<n; i++){
        a += text_line_boxes[i][0]*text_line_boxes[i][0];
        b += text_line_boxes[i][0];
        c += text_line_boxes[i][0]*text_line_boxes[i][3];
        d += text_line_boxes[i][3];
    }
    float k = (c*n-b*d)/(a*n-b*b);
    float bb = (a*d-c*b)/(a*n-b*b);
    //cout<<"2 "<<k<<" "<<bb<<endl;
    float p = k*x + bb;
    return p;
}


/*void threshold(vector< vector<float> >& coords, int min_, int max_)
{
    
}


void clip_boxes(vector< vector<float> >& boxes, int h, int w)
{
    boxes = threshold(, 0, w-1);
    boxes = threshold(, 0, h-1);
}*/


vector< vector<float> > get_text_lines(vector< vector<float> >& text_proposals, vector<float>& scores, int h, int w)
{
	//graph builder
    vector< vector<int> > tp_groups;
    tp_groups = group_text_proposals(text_proposals, scores, h, w);


    /*for(int i=0; i<tp_groups.size(); i++){
        for(int j=0; j<tp_groups[i].size(); j++){
            cout<<tp_groups[i][j]<<" ";
        }
        cout<<endl;
    }*/

    vector< vector<float> > text_lines(tp_groups.size(),vector<float>(5, 0));

    for(size_t i=0; i<tp_groups.size(); i++){
        vector< vector<float> > text_line_boxes(tp_groups[i].size());
        for(size_t j=0; j<tp_groups[i].size(); j++){
            text_line_boxes[j] = text_proposals[tp_groups[i][j]];
        }

        float x0 = text_line_boxes[0][0];
        float x1 = text_line_boxes[0][2];
        for(size_t k=1; k<tp_groups[i].size(); k++){
            if(text_line_boxes[k][0]<x0){
                x0 = text_line_boxes[k][0];
            }
            if(text_line_boxes[k][2]>x1){
                x1 = text_line_boxes[k][2];
            }
        }

        float offset = (text_line_boxes[0][2]-text_line_boxes[0][0])*0.5;

        float lt_y, rt_y, lb_y, rb_y;
        lt_y = fit_y1(text_line_boxes, x0+offset);
        rt_y = fit_y1(text_line_boxes, x1-offset);
        lb_y = fit_y2(text_line_boxes, x0+offset);
        rb_y = fit_y2(text_line_boxes, x1-offset);
        //cout<<lt_y<<" "<<rt_y<<" "<<lb_y<<" "<<rb_y<<endl;

        //average score
        float score = 0;
        for(size_t p=0; p<tp_groups[i].size(); p++){
            score += scores[tp_groups[i][p]];
        }
        score = score/tp_groups[i].size();

        text_lines[i][0] = x0;
        text_lines[i][1] = min(lt_y, rt_y);
        text_lines[i][2] = x1;
        text_lines[i][3] = max(lb_y, rb_y);
        text_lines[i][4] = score;
    }

    //clib_boxes(text_lines, h, w);
    /*for(int i=0; i<text_lines.size(); i++){
        for(int j=0; j<text_lines[i].size(); j++){
            cout<<text_lines[i][j]<<" ";
        }
        cout<<endl;
    }*/ 

	return text_lines;
}


vector<int> filter_boxes(vector< vector<float> >& boxes)
{
    vector<int> keep_inds;
    for(size_t i=0; i<boxes.size(); i++){
        float height = boxes[i][3] - boxes[i][1] + 1;
        float width = boxes[i][2] - boxes[i][0] + 1;
        float score = boxes[i][4];
        if(width/height>MIN_RATIO && score>LINE_MIN_SCORE && width>TEXT_PROPOSALS_WIDTH*MIN_NUM_PROPOSALS){
            keep_inds.push_back(i);
        }
    }
    return keep_inds;
}

